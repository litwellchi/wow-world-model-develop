#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import torch
from PIL import Image
import gradio as gr
from pathlib import Path
import cv2
import argparse
import threading
import sys
import concurrent.futures
from typing import Optional

sys.path.insert(0, "ditmodels/wow_wan/DiffSynth-Studio")
from diffsynth import ModelManager, WanVideoPipeline, save_video, wan_video
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from diffsynth.models.wan_video_vace import VaceWanModel
from diffsynth.models.wan_video_motion_controller import WanMotionControllerModel

def wow_model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    x: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    vace_context=None,
    vace_scale=1.0,
    tea_cache: wan_video.TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    **kwargs,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
            get_sp_group,
        )

    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)
        if hasattr(dit, "img_emb"):
            clip_embdding = dit.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

    x = dit.patchify(x)
    f, h, w = x.shape[2], x.shape[3], x.shape[4]

    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)

    tea_cache_update = tea_cache.check(dit, x, t_mod) if tea_cache is not None else False

    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)

    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        for block_id, block in enumerate(dit.blocks):
            x = block(x, context, t_mod, freqs)
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                x = x + vace_hints[vace.vace_layers_mapping[block_id]] * vace_scale
        if tea_cache is not None:
            tea_cache.store(x)

    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    x = dit.unpatchify(x, (f, h, w))
    return x

wan_video.model_fn_wan_video = wow_model_fn_wan_video

torch.serialization.add_safe_globals(["set", "OrderedDict", "builtins.set"])

FIXED_CHECKPOINT_PATH = "ditmodels/checkpoints/wan-epoch=95-train_loss=0.0265.ckpt"

def extract_first_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read video: {video_path}")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

def build_pipeline(gpu_id=0, checkpoint_path=None):
    device = f"cuda:{gpu_id}"
    mm = ModelManager(device="cpu")
    mm.load_models(
        ["ditmodels/wow_wan/models--Wan-AI--Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float32,
    )
    dit_paths = [
        f"ditmodels/wow_wan/models--Wan-AI--Wan2.1-I2V-14B-480P/diffusion_pytorch_model-0000{i}-of-00007.safetensors"
        for i in range(1, 8)
    ]
    mm.load_models(
        [
            dit_paths,
            "ditmodels/wow_wan/models--Wan-AI--Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
            "ditmodels/wow_wan/models--Wan-AI--Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16,
    )
    if checkpoint_path:
        checkpoint_file = Path(checkpoint_path) / "checkpoint" / "mp_rank_00_model_states.pt"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Model file does not exist: {checkpoint_file}")
        state_dict = torch.load(str(checkpoint_file), map_location="cpu")
        dit_model = mm.fetch_model("wan_video_dit")
        if dit_model is not None:
            dit_model.load_state_dict(state_dict, strict=False)
    pipe = WanVideoPipeline.from_model_manager(mm, torch_dtype=torch.bfloat16, device=device)
    pipe.enable_vram_management(num_persistent_param_in_dit=60 * 10**9)
    return pipe

PIPES = {}
LOCKS = {}
EXECUTORS = {}
FUTURES = {}
STOP_FLAGS = {}

def generate_video(prompt, input_file, gpu_id, steps=30, seed=42, tiled=True, num_frames=80, width=832, height=480, stop_event=None):
    if isinstance(gpu_id, (str,)):
        try:
            gpu_id = int(float(gpu_id))
        except Exception:
            gpu_id = None
    try:
        gpu_id = int(gpu_id)
    except Exception:
        gpu_id = None

    if not prompt or input_file is None:
        return "Prompt and input image are required", None

    if not PIPES:
        return "Model(s) not loaded, please check startup logs", None
    if stop_event is not None and stop_event.is_set():
        return "Stopped by user.", None

    if gpu_id in PIPES:
        pipe = PIPES[gpu_id]
        lock = LOCKS[gpu_id]
    else:
        fallback_gpu = next(iter(PIPES))
        pipe = PIPES[fallback_gpu]
        lock = LOCKS[fallback_gpu]

    if hasattr(input_file, "name") and input_file.name.lower().endswith(".mp4"):
        input_image = extract_first_frame(input_file.name)
    else:
        input_image = Image.open(input_file).convert("RGB")

    with lock:
        video = pipe(
            prompt=prompt,
            negative_prompt="low quality, distorted, ugly, bad anatomy",
            input_image=input_image,
            num_inference_steps=steps,
            seed=seed,
            tiled=tiled,
            num_frames=num_frames,
            width=width,
            height=height,
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            output_path = tmpfile.name
        save_video(video, output_path, fps=16, quality=5)
    return "Generation successful!", output_path

def build_interface():
    with gr.Blocks(title="Wow Video Generation") as demo:
        gr.Markdown("## 🎬 Wow Video Generation — Multi-GPU Panels")
        gr.Markdown("Each GPU has its own panel. You can submit multiple tasks in parallel (different GPUs run concurrently, same GPU runs sequentially).")
        if not PIPES:
            gr.Markdown("**No models loaded. Please specify --gpus at startup.**")
            return demo
        with gr.Row():
            for g in sorted(PIPES.keys()):
                with gr.Column():
                    gr.Markdown(f"### GPU {g}")
                    inp = gr.File(label="Upload image or video (.jpg/.png/.mp4)", interactive=True)
                    prm = gr.Textbox(label="Text prompt", placeholder="Describe the scene...")
                    steps = gr.Slider(1, 100, value=30, step=1, label="Inference steps")
                    seed = gr.Number(label="Random seed", value=42, precision=0)
                    tiled = gr.Checkbox(label="Use Tiled mode", value=True)
                    num_frames = gr.Slider(1, 120, value=80, step=1, label="Number of frames")
                    width = gr.Number(label="Width", value=832, precision=0)
                    height = gr.Number(label="Height", value=480, precision=0)
                    gen_btn = gr.Button(f"🚀 Generate on GPU {g}")
                    stop_btn = gr.Button(f"🛑 Stop on GPU {g}")
                    status = gr.Textbox(label="Status")
                    out_vid = gr.Video(label="Generated video", format="mp4")

                    if g not in EXECUTORS:
                        EXECUTORS[g] = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    if g not in STOP_FLAGS:
                        STOP_FLAGS[g] = threading.Event()

                    def start_task(prompt, input_file, steps_v, seed_v, tiled_v, num_frames_v, width_v, height_v, status, out_vid, gpu_id=g):
                        STOP_FLAGS[gpu_id].set()
                        status = ""
                        out_vid = None
                        STOP_FLAGS[gpu_id] = threading.Event()
                        result_status, result_video = generate_video(
                            prompt, input_file, gpu_id,
                            int(steps_v), int(seed_v), bool(tiled_v), int(num_frames_v),
                            int(width_v), int(height_v),
                            STOP_FLAGS[gpu_id]
                        )
                        return result_status, result_video

                    def stop_task():
                        STOP_FLAGS[g].set()
                        return "Stopped.", None

                    gen_btn.click(
                        start_task,
                        inputs=[prm, inp, steps, seed, tiled, num_frames, width, height, status, out_vid],
                        outputs=[status, out_vid],
                    )
                    stop_btn.click(
                        stop_task,
                        inputs=[],
                        outputs=[status, out_vid],
                    )
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7620, help="Gradio server port")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs to preload")
    parser.add_argument("--share", action="store_true", help="Whether to create a public link")
    args = parser.parse_args()

    gpu_list = []
    for x in args.gpus.split(","):
        x = x.strip()
        if x == "":
            continue
        try:
            gpu_list.append(int(x))
        except ValueError:
            continue
    if not gpu_list:
        gpu_list = [0]

    print(f"🚀 Loading models to GPUs: {gpu_list}, please wait...")

    for g in gpu_list:
        try:
            p = build_pipeline(gpu_id=g, checkpoint_path=FIXED_CHECKPOINT_PATH)
            PIPES[g] = p
            LOCKS[g] = threading.Lock()
            print(f"✅ Loaded pipe on cuda:{g}")
        except Exception as e:
            print(f"❌ Failed to load model on cuda:{g}: {e}")

    if not PIPES:
        raise RuntimeError("No models loaded. Exiting.")

    print("✅ Models loaded. Launching Gradio interface...")

    demo = build_interface()
    demo.launch(server_port=args.port, share=args.share)