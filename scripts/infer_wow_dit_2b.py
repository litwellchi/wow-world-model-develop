#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for generating videos using the WoW-Dit model.
"""

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

# logging setting 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# path configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
wowdit_dir = os.path.join(project_root, "dit_models", "wow-dit-2b")
video2world_path = os.path.join(wowdit_dir, "video2world.py")
ditckpt_path = os.path.join(wowdit_dir, "checkpoints", "wow_dit_2b.pt")
BASE_CMD = [
    "python", video2world_path,
    "--resolution", "720",
    "--dit_path", ditckpt_path,
    "--num_conditional_frames", "1",
    "--guidance", "7",
    "--seed", "42",
    "--disable_prompt_refiner",
    "--disable_guardrail",
    "--fps", "16"
]

INPUT_JSONL_PATH = os.path.join(project_root, "benchmark_samples", "generated_descriptions.jsonl")
OUTPUT_DIR = project_root
OUTPUT_VIDEO_DIR = os.path.join(OUTPUT_DIR, "generated_videos_wowdit2b")
RESULTS_FILE = os.path.join(OUTPUT_VIDEO_DIR, "wowdit2b_results.jsonl")
NUM_GPUS = 8
GPU_IDS = list(range(NUM_GPUS))


def chunk_list(lst, n):
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


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


def process_videos_single_gpu(videos_data, gpu_id, results_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    results = []
    
    for data in tqdm.tqdm(videos_data, desc=f"GPU {gpu_id}"):
        try:

            index = data["index"]
            image_path = data["image"]
            generated_description = data["generated_description"]
            original_lang = data.get("original_lang", "")
            
            actual_image_path = find_existing_image(image_path)
            if not actual_image_path:
                logger.error(f"Image not found: {image_path}")
                result_entry = {
                    "index": index,
                    "output_path": "",
                    "image_path": image_path,
                    "generated_description": generated_description,
                    "original_lang": original_lang,
                    "status": "failed",
                    "error": f"Image not found: {image_path}"
                }
                results.append(result_entry)
                continue
            

            os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
            

            index_int = int(index) if isinstance(index, str) else index
            video_name = f"{index_int:08d}.mp4"
            save_path = os.path.join(OUTPUT_VIDEO_DIR, video_name)
            

            dynamic_seed = int(time.time() * 1000000) % (2**32)
            

            cmd = BASE_CMD[:-2] + [  
                "--input_path", actual_image_path,
                "--save_path", save_path,
                "--prompt", generated_description,
                "--seed", str(dynamic_seed)
            ]
            

            env = os.environ.copy()
            env["CUDA_HOME"] = os.getenv("CONDA_PREFIX", "")
            env["PYTHONPATH"] = f"{wowdit_dir}:{project_root}:{script_dir}"

            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            

            logger.info(f"GPU {gpu_id}: Processing index {index} (Save as: {save_path})")
            result = subprocess.run(
                cmd,
                env=env,
                cwd=wowdit_dir,  
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if os.path.exists(save_path):
                result_entry = {
                    "index": index,
                    "output_path": save_path,
                    "image_path": actual_image_path,
                    "generated_description": generated_description,
                    "original_lang": original_lang,
                    "status": "success"
                }
                logger.info(f"GPU {gpu_id}: Success for index {index}")
            else:
                result_entry = {
                    "index": index,
                    "output_path": "",
                    "image_path": actual_image_path,
                    "generated_description": generated_description,
                    "original_lang": original_lang,
                    "status": "failed",
                    "error": "Output video file not generated"
                }
                logger.error(f"GPU {gpu_id}: Failed for index {index} - output file not found")
            
            results.append(result_entry)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"GPU {gpu_id}: Command failed for index {data.get('index', 'unknown')}: {e}")
            logger.error(f"GPU {gpu_id}: stderr: {e.stderr}")
            result_entry = {
                "index": data.get("index", -1),
                "output_path": "",
                "image_path": data.get("image_path", ""),
                "generated_description": data.get("generated_description", ""),
                "original_lang": data.get("original_lang", ""),
                "status": "failed",
                "error": f"Command failed: {str(e)}"
            }
            results.append(result_entry)
        except Exception as e:
            logger.error(f"GPU {gpu_id}: Unexpected error for index {data.get('index', 'unknown')}: {e}")
            result_entry = {
                "index": data.get("index", -1),
                "output_path": "",
                "image_path": data.get("image_path", ""),
                "generated_description": data.get("generated_description", ""),
                "original_lang": data.get("original_lang", ""),
                "status": "failed",
                "error": f"Unexpected error: {str(e)}"
            }
            results.append(result_entry)
    
    with multiprocessing.Lock():
        with open(results_file, 'a', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info(f"GPU {gpu_id}: Completed processing {len(videos_data)} videos")


def load_input_data():

    logger.info(f"Loading input data from: {INPUT_JSONL_PATH}")
    
    if not os.path.exists(INPUT_JSONL_PATH):
        logger.error(f"Input file not found: {INPUT_JSONL_PATH}")
        return []
    
    videos_data = []
    try:
        with open(INPUT_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        if "index" in item and "image" in item and "generated_description" in item:
                            videos_data.append(item)
                        else:
                            logger.warning(f"Line {line_num}: Missing required fields, skipping")
                    except json.JSONDecodeError as e:
                        logger.error(f"Line {line_num}: JSON decode error: {e}")
        
        logger.info(f"Loaded {len(videos_data)} valid entries")
        return videos_data
    
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        return []


def main():
    logger.info("Starting Cosmos2 initial video generation...")
    logger.info(f"Input file: {INPUT_JSONL_PATH}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Results file: {RESULTS_FILE}")
    logger.info(f"Using {NUM_GPUS} GPUs: {GPU_IDS}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
    
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
        logger.info(f"Removed existing results file: {RESULTS_FILE}")
    
    videos_data = load_input_data()
    if not videos_data:
        logger.error("No valid input data found, exiting")
        return
    
    logger.info(f"Processing {len(videos_data)} videos...")
    
    video_chunks = list(chunk_list(videos_data, NUM_GPUS))
    
    processes = []
    for gpu_id, chunk in enumerate(video_chunks):
        if chunk: 
            logger.info(f"GPU {gpu_id}: Assigned {len(chunk)} videos")
            p = Process(target=process_videos_single_gpu, args=(chunk, gpu_id, RESULTS_FILE))
            p.start()
            processes.append(p)
        else:
            logger.info(f"GPU {gpu_id}: No videos assigned")
    
    for p in processes:
        p.join()
    
    logger.info("All GPU processes completed")
    
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            results = [json.loads(line.strip()) for line in f if line.strip()]
        
        def get_sort_key(item):
            index = item.get('index', 0)
            if isinstance(index, str):
                try:
                    return int(index)
                except ValueError:
                    return index
            return index
        
        results.sort(key=get_sort_key)
        
        logger.info("Sorting results by index...")
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        success_count = sum(1 for r in results if r.get('status') == 'success')
        failed_count = len(results) - success_count
        
        logger.info(f"Generation completed!")
        logger.info(f"Total: {len(results)}")
        logger.info(f"Success: {success_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Success rate: {success_count/len(results)*100:.1f}%")
        logger.info(f"Results saved to: {RESULTS_FILE}")
    else:
        logger.error("No results file found")


if __name__ == "__main__":
    main()