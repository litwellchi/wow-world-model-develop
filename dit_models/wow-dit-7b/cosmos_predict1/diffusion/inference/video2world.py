# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import torch

from cosmos_predict1.diffusion.inference.inference_utils import (
    add_common_arguments,
    check_input_frames,
    get_input_sizes,
    validate_args,
)
from cosmos_predict1.diffusion.inference.world_generation_pipeline import DiffusionVideo2WorldGenerationPipeline
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.io import read_prompts_from_file, save_video

torch.enable_grad(False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)

    # Add video2world specific arguments
    parser.add_argument(
        "--diffusion_transformer_dir",
        type=str,
        default="Cosmos-Predict1-7B-Video2World",
        help="DiT model weights directory name relative to checkpoint_dir",
        choices=[
            "Cosmos-Predict1-7B-Video2World",
            "Cosmos-Predict1-14B-Video2World",
            "Cosmos-Predict1-7B-Video2World_post-trained",
            "Cosmos-Predict1-7B-Video2World-sft",
            "Cosmos-Predict1-7B-Video2World_post-trained-4gpu_80gb",
            "Cosmos-Predict1-7B-Video2World_post-trained-8gpu_40gb",
            "Cosmos-Predict1-7B-Video2World_post-trained-4gpu_40gb",
            "Cosmos-Predict1-7B-Video2World_post-trained-lora",
            "Cosmos-Predict1-14B-Video2World_post-trained",
            "WoWDiT-7B-WorldModel",
        ],
    )
    parser.add_argument(
        "--prompt_upsampler_dir",
        type=str,
        default="Pixtral-12B",
        help="Prompt upsampler weights directory relative to checkpoint_dir",
    )
    parser.add_argument(
        "--input_image_or_video_path",
        type=str,
        help="Input video/image path for generating a single video",
    )
    parser.add_argument(
        "--num_input_frames",
        type=int,
        default=1,
        help="Number of input frames for video2world prediction",
        choices=[1, 9],
    )

    return parser.parse_args()


def demo(args):
    """Run video-to-world generation demo.

    This function handles the main video-to-world generation pipeline, including:
    - Setting up the random seed for reproducibility
    - Initializing the generation pipeline with the provided configuration
    - Processing single or multiple prompts/images/videos from input
    - Generating videos from prompts and images/videos
    - Saving the generated videos and corresponding prompts to disk

    Args:
        cfg (argparse.Namespace): Configuration namespace containing:
            - Model configuration (checkpoint paths, model settings)
            - Generation parameters (guidance, steps, dimensions)
            - Input/output settings (prompts/images/videos, save paths)
            - Performance options (model offloading settings)

    The function will save:
        - Generated MP4 video files
        - Text files containing the processed prompts

    If guardrails block the generation, a critical log message is displayed
    and the function continues to the next prompt if available.
    """
    misc.set_random_seed(args.seed)
    inference_type = "video2world"
    validate_args(args, inference_type)

    if args.num_gpus > 1:
        from megatron.core import parallel_state

        from cosmos_predict1.utils import distributed

        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        process_group = parallel_state.get_context_parallel_group()

    # Initialize video2world generation model pipeline
    pipeline = DiffusionVideo2WorldGenerationPipeline(
        inference_type=inference_type,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.diffusion_transformer_dir,
        prompt_upsampler_dir=args.prompt_upsampler_dir,
        enable_prompt_upsampler=not args.disable_prompt_upsampler,
        offload_network=args.offload_diffusion_transformer,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_prompt_upsampler=args.offload_prompt_upsampler,
        offload_guardrail_models=args.offload_guardrail_models,
        disable_guardrail=args.disable_guardrail,
        guidance=args.guidance,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_video_frames=args.num_video_frames,
        seed=args.seed,
        num_input_frames=args.num_input_frames,
    )

    if args.num_gpus > 1:
        pipeline.model.net.enable_context_parallel(process_group)

    # Handle multiple prompts if prompt file is provided
    if args.batch_input_path:
        log.info(f"Reading batch inputs from path: {args.batch_input_path}")
        prompts = read_prompts_from_file(args.batch_input_path)
    else:
        # Single prompt case
        prompts = [{"prompt": args.prompt, "visual_input": args.input_image_or_video_path}]

    os.makedirs(args.video_save_folder, exist_ok=True)
    for i, input_dict in enumerate(prompts):
        current_prompt = input_dict.get("prompt", None)
        if current_prompt is None and args.disable_prompt_upsampler:
            log.critical("Prompt is missing, skipping world generation.")
            continue
        current_image_or_video_path = input_dict.get("visual_input", None)
        if current_image_or_video_path is None:
            log.critical("Visual input is missing, skipping world generation.")
            continue

        # Check input frames
        if not check_input_frames(current_image_or_video_path, args.num_input_frames):
            continue
        log.warning("Visual input is provided, overriding --height and --width arguments.")
        # OlD
        # args.height, args.width = get_input_sizes(current_image_or_video_path)
        # New
        current_height, current_width = get_input_sizes(current_image_or_video_path)
        if args.height is not None and args.width is not None:
            if args.height != current_height or args.width != current_width:
                log.warning("Visual input's size is different from --height and --width arguments.")
                log.warning("manually adjusting model state_shape...")
                pipeline.model.state_shape = [pipeline.model.state_shape[0], pipeline.model.state_shape[1], args.height//pipeline.model.tokenizer.spatial_compression_factor, args.width//pipeline.model.tokenizer.spatial_compression_factor]
        else:
            args.height, args.width = current_height, current_width
        
        # Generate video
        generated_output = pipeline.generate(
            prompt=current_prompt,
            image_or_video_path=current_image_or_video_path,
            negative_prompt=args.negative_prompt,
        )
        if generated_output is None:
            log.critical("Guardrail blocked video2world generation.")
            continue
        video, prompt = generated_output

        if args.batch_input_path:
            video_save_path = os.path.join(args.video_save_folder, f"{i}.mp4")
            prompt_save_path = os.path.join(args.video_save_folder, f"{i}.txt")
        else:
            video_save_path = os.path.join(args.video_save_folder, f"{args.video_save_name}.mp4")
            prompt_save_path = os.path.join(args.video_save_folder, f"{args.video_save_name}.txt")

        # Save video
        save_video(
            video=video,
            fps=args.fps,
            H=args.height,
            W=args.width,
            video_save_quality=5,
            video_save_path=video_save_path,
        )

        # Save prompt to text file alongside video
        with open(prompt_save_path, "wb") as f:
            f.write(prompt.encode("utf-8"))

        log.info(f"Saved video to {video_save_path}")
        log.info(f"Saved prompt to {prompt_save_path}")

    # clean up properly
    if args.num_gpus > 1:
        parallel_state.destroy_model_parallel()
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_arguments()
    demo(args)
