#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import subprocess
import os
import logging
from datetime import datetime
import sys
import time
from pathlib import Path
import traceback
from PIL import Image
import multiprocessing
from multiprocessing import Process
import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
wowdit_dir = os.path.join(project_root, "dit_models", "wow-dit-7b")
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

BASE_CMD = [
    "python", os.path.join(wowdit_dir, "cosmos_predict1/diffusion/inference/video2world.py"),
    "--checkpoint_dir", os.path.join(wowdit_dir, "checkpoints/"),
    "--diffusion_transformer_dir", "Cosmos-Predict1-7B-Video2World_post-trained",
    "--video_save_folder", os.path.join(repo_root, "generated_videos_wowdit7b"),
    "--num_input_frames", "1",
    "--disable_prompt_upsampler",
    "--disable_guardrail",
    "--offload_diffusion_transformer",
    "--offload_text_encoder_model",
    "--height", "720",
    "--width", "1024"
]


TRAINED_CKPT_PATH = os.path.join(wowdit_dir, "checkpoints/wow_dit_7b.pt")


INPUT_JSONL_PATH = os.path.join(repo_root, "benchmark_samples", "generated_descriptions.jsonl")
OUTPUT_DIR = os.path.join(repo_root, "generated_videos_wowdit7b")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "video_generation_results.jsonl")


NUM_GPUS = 8
GPU_IDS = list(range(NUM_GPUS))


def setup_config():
    # Create a symbolic link for the trained checkpoint path
    symlink_target = os.path.join(BASE_CMD[3], BASE_CMD[5], 'model.pt')
    if not os.path.exists(symlink_target):
        os.makedirs(os.path.dirname(symlink_target), exist_ok=True)
        os.symlink(TRAINED_CKPT_PATH, symlink_target)
        print(f"Created symbolic link: {symlink_target} -> {TRAINED_CKPT_PATH}")
    else:
        os.remove(symlink_target)
        os.symlink(TRAINED_CKPT_PATH, symlink_target)
        print(f"Symbolic link already exists: {symlink_target}, overwrite it")


def find_existing_image(image_path):

    if os.path.exists(image_path):
        return image_path

    base_path, ext = os.path.splitext(image_path)

    possible_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']

    for new_ext in possible_extensions:
        if new_ext.lower() != ext.lower():
            new_path = base_path + new_ext
            if os.path.exists(new_path):
                return new_path
    
    return None


def prepare_prompt(description: str) -> str:

    return description


def chunk_list(lst, n):
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def process_videos(videos_data, gpu_id, results_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    for i, data in enumerate(tqdm.tqdm(videos_data, desc=f"GPU {gpu_id}")):
        try:
            if 'index' in data:
                video_id = data['index']
            elif 'unique_id' in data:
                video_id = data['unique_id'].replace('/', '_')
            else:
                video_id = f"video_{i:05d}"
            
            if isinstance(video_id, str):
                import re
                numbers = re.findall(r'\d+', video_id)
                if numbers:
                    video_id_int = int(numbers[-1])
                else:
                    video_id_int = i
            else:
                video_id_int = int(video_id)
            
            video_name = f"{video_id_int:08d}.mp4"
            output_path = os.path.join(OUTPUT_DIR, video_name)
            

            result = generate_single_video(data, output_path)

            if result['success']:
                result_item = {
                    'index': video_id,
                    'output_path': result['output_path'],
                    'image_path': result.get('image_path', data.get('image_path', '')),
                    'generated_description': data.get('generated_description', ''),
                    'original_lang': data.get('original_lang', ''),
                    'status': 'success',
                }
                logger.info(f"GPU {gpu_id} - Success generate video: {video_id}")
            else:
                result_item = {
                    'index': video_id,
                    'output_path': '',
                    'image_path': data.get('image_path', ''),
                    'generated_description': data.get('generated_description', ''),
                    'original_lang': data.get('original_lang', ''),
                    'status': 'failed',
                    'error': result['error'],
                }
                logger.error(f"GPU {gpu_id} - generate fail: {video_id} - {result['error']}")
            
            with open(results_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"GPU {gpu_id} - process video error: {str(e)}")
            result_item = {
                'index': video_id if 'video_id' in locals() else f"error_{i}",
                'output_path': '',
                'image_path': data.get('image_path', ''),
                'generated_description': data.get('generated_description', ''),
                'original_lang': data.get('original_lang', ''),
                'status': 'failed',
                'error': str(e),
            }
            with open(results_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_item, ensure_ascii=False) + '\n')


def generate_single_video(input_data: dict, output_path: str) -> dict:
    try:
        if 'generated_description' in input_data:
            description = input_data['generated_description']
        else:
            logger.error("lost description in input data")
            return {
                'success': False,
                'error': 'lost description in input data'
            }
        
        original_lang = input_data.get('original_lang', '')
        
        image_path = input_data.get('image', '')
        if not image_path:
            logger.error("lost image path in input data")
            return {
                'success': False,
                'error': 'lost image path in input data'
            }

        actual_image_path = find_existing_image(image_path)
        if actual_image_path is None:
            logger.error(f"image file non-exist: {image_path}")
            return {
                'success': False,
                'error': f'image file non-exist: {image_path}'
            }
        
        if actual_image_path != image_path:
            logger.info(f"image format autofix: {image_path} -> {actual_image_path}")
        
        prompt = prepare_prompt(description)
        
        video_name = os.path.basename(output_path).replace('.mp4', '')

        seed = int(time.time() * 1000) % (2**32)
        

        cmd = BASE_CMD + [
            "--input_image_or_video_path", actual_image_path,
            "--video_save_name", video_name,
            "--prompt", prompt,
            "--seed", str(seed)
        ]
        

        env = os.environ.copy()
        env["CUDA_HOME"] = os.getenv("CONDA_PREFIX", "")

        # cosmos1_dir = wowdit_dir
        env["PYTHONPATH"] = f"{wowdit_dir}:{repo_root}:{os.getcwd()}"
        
        logger.info(f"Gen Video: {video_name}")
        logger.info(f"Prompt: {prompt[:100]}...")
        
        result = subprocess.run(
            cmd,
            env=env,
            cwd=wowdit_dir,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True
        )
        

        if os.path.exists(output_path):
            txt_file = output_path.replace('.mp4', '.txt')
            if os.path.exists(txt_file):
                os.remove(txt_file)
                logger.debug(f"delete text: {txt_file}")
            
            logger.info(f"Success Gen Video: {output_path}")
            return {
                'success': True,
                'output_path': output_path,
                'image_path': actual_image_path,
                'original_lang': original_lang,
                'prompt': prompt,
                'seed': seed
            }
        else:
            logger.error(f"Output File Unfound: {output_path}")
            return {
                'success': False,
                'error': 'Output File Unfound'
            }
            
    except subprocess.CalledProcessError as e:
        logger.error(f"video gen fail: {str(e)}")
        return {
            'success': False,
            'error': f'video gen fail: {str(e)}'
        }
    except Exception as e:
        logger.error(f"video gen error: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e)
        }


def sort_results_file(file_path):
    if not os.path.exists(file_path):
        return

    try:
        results = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"can not parse line: {line}")
        
        for r in results:
            if 'index' in r and isinstance(r['index'], str):
                try:
                    r['index'] = int(r['index'])
                except ValueError:
                    logger.warning(f"index is not a number: {r['index']}")
                    r['index'] = float('inf')
        

        results.sort(key=lambda x: x.get('index', float('inf')))
        

        with open(file_path, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Sorted {file_path} ")

    except Exception as e:
        logger.error(f"Error when sorting index: {e}")


def process_dataset():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    setup_config()
    
    logger.info(f"Read Input: {INPUT_JSONL_PATH}")
    
    if not os.path.exists(INPUT_JSONL_PATH):
        logger.error(f"Input is not exist: {INPUT_JSONL_PATH}")
        return
    
    data = []
    with open(INPUT_JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if 'index' in obj:
                obj['index'] = int(obj['index'])
            data.append(obj)
    
    logger.info(f"Loaded {len(data)} samples")
    

    total = len(data)
    start_time = time.time()
    
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    video_chunks = list(chunk_list(data, NUM_GPUS))

    processes = []
    for gpu_id, chunk in enumerate(video_chunks):
        if chunk: 
            p = Process(target=process_videos, args=(chunk, gpu_id, RESULTS_FILE))
            p.start()
            processes.append(p)
            logger.info(f"GPU {gpu_id} Process，in total {len(chunk)} items")
    
    for p in processes:
        p.join()

    sort_results_file(RESULTS_FILE)

    results = []
    success_count = 0
    failed_count = 0
    
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        result = json.loads(line)
                        results.append(result)
                        if result.get('status') == 'success':
                            success_count += 1
                        else:
                            failed_count += 1
                    except json.JSONDecodeError:
                        failed_count += 1

    txt_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')]
    if txt_files:
        logger.info(f"deleta {len(txt_files)} txt files...")
        for txt_file in txt_files:
            txt_path = os.path.join(OUTPUT_DIR, txt_file)
            if os.path.exists(txt_path):
                os.remove(txt_path)
    
    total_time = time.time() - start_time
    logger.info("\n" + "="*60)
    logger.info("Gen Done！")
    logger.info(f" {total} videos")
    success_rate = success_count/total*100 if total > 0 else 0
    failed_rate = failed_count/total*100 if total > 0 else 0
    logger.info(f"Success: {success_count} ({success_rate:.1f}%)")
    logger.info(f"Fail: {failed_count} ({failed_rate:.1f}%)")
    logger.info(f"Total Time: {total_time/60:.1f} min")
    logger.info(f"Average: {total_time/total:.1f} s/video" if total > 0 else "N/A")
    logger.info(f"Results: {RESULTS_FILE}")
    logger.info("="*60)


if __name__ == "__main__":
    logger.info("Starting initial video generation with multi-GPU support...")
    try:
        multiprocessing.set_start_method('spawn', force=True)
        process_dataset()
    except KeyboardInterrupt:
        logger.info("\nUser interrupted the process.")
    except Exception as e:
        logger.error(f"code run error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    logger.info("Done! Initial video generation completed.")
