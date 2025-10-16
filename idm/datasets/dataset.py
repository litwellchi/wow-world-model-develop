import os
import glob
import json
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import sys

# Ensure Video-Depth-Anything is importable. Use VDA_PATH env var or a generic placeholder.
vda_path = os.environ.get("VDA_PATH", "/path/to/Video-Depth-Anything")
if vda_path not in sys.path:
    sys.path.insert(0, vda_path)

from run import VDAFrameDepthInferencer


class RLBenchDataset(Dataset):
    def __init__(self, rootdir, transform=None):
        self.rootdir = rootdir
        self.actions = []
        self.image_pairs = []
        self.texts = []
        self.text_map = None

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        self.load_data()
        print("Loaded Data Nums:", self.__len__())

    def load_data(self):
        map_path = os.path.join(self.rootdir, "tasks.json")
        with open(map_path, "r") as map_file:
            self.text_map = json.load(map_file)

        for episode in os.listdir(self.rootdir):
            episode_dir = os.path.join(self.rootdir, episode)
            if not os.path.isdir(episode_dir):
                continue

            images = sorted(glob.glob(os.path.join(episode_dir, "images", "*.png")))
            actions = np.load(os.path.join(episode_dir, "actions.npy"))
            texts_f = open(os.path.join(episode_dir, "text.txt"), "r")
            texts_lines = texts_f.readlines()

            for i in range(1, len(images)):
                self.image_pairs.append([images[i - 1], images[i]])
                self.actions.append(actions[i - 1])
                self.texts.append(texts_lines[0])

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        images_path = self.image_pairs[idx]
        tensor_imgs = []
        for image_path in images_path:
            img = Image.open(image_path).convert('RGB')
            tensor_img = self.transform(img)
            tensor_imgs.append(tensor_img)
        img_pair = torch.stack(tensor_imgs, 0)

        action = torch.tensor(self.actions[idx], dtype=torch.float32)
        text = self.texts[idx].strip()
        text_id = torch.tensor(self.text_map[text])
        return img_pair, action, text_id




class RLBenchHDF5Dataset(Dataset):
    def __init__(self, rootdir=None, image_size=(128, 128), max_text_tokens=100):
        # rootdir can be provided or set via HDF5_ROOT environment variable
        self.rootdir = rootdir or os.environ.get("HDF5_ROOT", "/path/to/hdf5_root")
        self.image_size = image_size
        self.max_text_tokens = max_text_tokens
        
        self.hdf5_files = []
        self.actions = []
        self.image_indices = []
        self.texts = []
        self.text_map = {}
        self.text_counter = 0
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
        ])
        
        self.load_data()
        print("Loaded Data Nums:", self.__len__())
        
    def load_data(self):
        # Collect all hdf5 files
        for file in os.listdir(self.rootdir):
            if file.endswith('.hdf5'):
                hdf5_path = os.path.join(self.rootdir, file)
                self.hdf5_files.append(hdf5_path)
        
        # Create data indices for each hdf5 file
        for hdf5_path in self.hdf5_files:
            with h5py.File(hdf5_path, 'r') as f:
                # Read text instruction
                text = f['text'][()].decode('utf-8') if isinstance(f['text'][()], bytes) else str(f['text'][()])
                
                # Create mapping for new texts
                if text not in self.text_map:
                    self.text_map[text] = self.text_counter
                    self.text_counter += 1
                
                # Read actions and image data
                actions = f['action'][:]  # shape: (T, 7)
                images = f['observations']['image'][:]  # shape: (T, 224, 224, 3)
                
                # Create samples for each timestep
                for i in range(1, len(actions)):
                    self.actions.append(actions[i-1])  # current action
                    self.image_indices.append((hdf5_path, i-1, i))  # (file path, previous frame idx, current frame idx)
                    self.texts.append(text)
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        hdf5_path, prev_idx, curr_idx = self.image_indices[idx]
        
        with h5py.File(hdf5_path, 'r') as f:
            # Read two consecutive frames
            prev_image = f['observations']['image'][prev_idx]  # (224, 224, 3)
            curr_image = f['observations']['image'][curr_idx]  # (224, 224, 3)
            
            # Convert to PIL images and apply transform
            prev_img = Image.fromarray(prev_image).convert('RGB')
            curr_img = Image.fromarray(curr_image).convert('RGB')
            
            prev_tensor = self.transform(prev_img)
            curr_tensor = self.transform(curr_img)
            
        img_pair = torch.stack([prev_tensor, curr_tensor], 0)  # (2, 3, H, W)
        
        action = torch.tensor(self.actions[idx], dtype=torch.float32)  # shape: (7,)
        text = self.texts[idx].strip()
        text_id = torch.tensor(self.text_map[text])
        
        return img_pair, action, text_id


class RLBenchFLowFeatureDataset(Dataset):
    def __init__(self, rootdir, dino_model, cotracker_model, dino_transform, image_size=(128, 128), rank=0, world_size=1):
        self.rootdir = rootdir
        self.dino = dino_model.eval().cuda() if dino_model is not None else None
        self.flow_tracker = cotracker_model.eval().cuda() if cotracker_model is not None else None
        self.dino_transform = dino_transform
        self.image_size = image_size
        self.rank = rank
        self.world_size = world_size

        self.text_map = {}
        self.text_counter = 0
        self.samples = []

        self._precompute_all_features()

    def _load_hdf5_files(self):
        hdf5_files = []
        for file in os.listdir(self.rootdir):
            if file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(self.rootdir, file))
        return sorted(hdf5_files)  # ensure consistent order

    def _process_images_batch(self, images_batch):
        """Batch process images to improve efficiency"""
        if len(images_batch) == 0:
            return [], []
        
        # Batch transform images
        img_tensors = []
        for img1, img2 in images_batch:
            img1_tensor = self.dino_transform(img1).unsqueeze(0)
            img2_tensor = self.dino_transform(img2).unsqueeze(0)
            img_tensors.extend([img1_tensor, img2_tensor])
        
        # Batch DINO inference
        batch_tensor = torch.cat(img_tensors, 0).cuda()  # (2N, 3, H, W)
        batch_feats = self.dino(batch_tensor)  # (2N, D)
        
        # Group features
        img_feats = []
        for i in range(0, len(batch_feats), 2):
            feat1 = batch_feats[i]
            feat2 = batch_feats[i + 1]
            img_feat = torch.cat([feat1, feat2], dim=-1).cpu()  # (2D,)
            img_feats.append(img_feat)
        
        return img_feats

    def _process_flow_batch(self, images_batch):
        """Batch process flow features - fix batching issues by processing one-by-one"""
        if len(images_batch) == 0:
            return []
        
        flow_feats = []
        resize_transform = transforms.Resize((224, 224))
        tensor_transform = transforms.ToTensor()
        
        # CoTracker batching has issues; process one-by-one but minimize repeated ops
        for img1, img2 in images_batch:
            img1_resize = resize_transform(img1)
            img2_resize = resize_transform(img2)
            img1_arr = tensor_transform(img1_resize).unsqueeze(0).cuda() * 255
            img2_arr = tensor_transform(img2_resize).unsqueeze(0).cuda() * 255
            input_video = torch.stack([img1_arr, img2_arr], dim=1)  # (1, 2, 3, H, W)
            input_video = input_video.repeat(1, 5, 1, 1, 1)  # (1, 5, 3, H, W)
            
            # Single inference
            pred_tracks, _ = self.flow_tracker(input_video, grid_size=20)  # ensure grid_size matches model expectation
            last_tracks = pred_tracks[0, -1].reshape(-1).cpu()  # (800,)
            flow_feats.append(last_tracks)
        
        return flow_feats

    @torch.no_grad()
    def _precompute_all_features(self):
        features_path = os.path.join(self.rootdir, "features.pth")
        
        # Only rank 0 performs preprocessing; other processes wait
        if self.rank == 0:
            if not os.path.exists(features_path):
                print(f"[Rank {self.rank}] Starting preprocessing...")
                hdf5_files = self._load_hdf5_files()
                batch_size = 16  # reduce batch size to avoid memory issues
                max_files_per_save = 1000  # save periodically to avoid OOM
                processed_files = 0

                for file_idx, hdf5_path in enumerate(tqdm(hdf5_files, desc="[Preprocessing HDF5 Files]")):
                    with h5py.File(hdf5_path, 'r') as f:
                        text = f['text'][()].decode('utf-8') if isinstance(f['text'][()], bytes) else str(f['text']())
                        if text not in self.text_map:
                            self.text_map[text] = self.text_counter
                            self.text_counter += 1
                        text_id = self.text_map[text]

                        actions = f['action'][:]  # (T, 7)
                        images = f['observations']['image'][:]  # (T, H, W, 3)
                        
                        # Prepare image pairs
                        image_pairs = []
                        actions_list = []
                        for i in range(1, len(actions)):
                            img1 = Image.fromarray(images[i - 1]).convert('RGB')
                            img2 = Image.fromarray(images[i]).convert('RGB')
                            image_pairs.append((img1, img2))
                            actions_list.append(actions[i - 1])
                        
                        # Batch processing
                        for batch_start in range(0, len(image_pairs), batch_size):
                            batch_end = min(batch_start + batch_size, len(image_pairs))
                            batch_pairs = image_pairs[batch_start:batch_end]
                            batch_actions = actions_list[batch_start:batch_end]
                            
                            # Batch process DINO features
                            if self.dino is not None:
                                img_feats = self._process_images_batch(batch_pairs)
                            else:
                                img_feats = [torch.zeros(2048) for _ in batch_pairs]  # placeholder
                            
                            # Batch process flow features
                            if self.flow_tracker is not None:
                                flow_feats = self._process_flow_batch(batch_pairs)
                            else:
                                flow_feats = [torch.zeros(800) for _ in batch_pairs]  # placeholder, matches grid_size=20
                            
                            # Save samples
                            for img_feat, flow_feat, action in zip(img_feats, flow_feats, batch_actions):
                                self.samples.append({
                                    'img_feat': img_feat,
                                    'flow_feat': flow_feat,
                                    'action': torch.tensor(action, dtype=torch.float32),
                                    'text_id': text_id
                                })
                    
                    processed_files += 1
                    
                    # Periodically save to avoid OOM and data loss
                    if processed_files % max_files_per_save == 0:
                        temp_save_data = {
                            'samples': self.samples,
                            'text_map': self.text_map,
                            'text_counter': self.text_counter
                        }
                        temp_path = features_path.replace('.pth', f'_temp_{processed_files}.pth')
                        torch.save(temp_save_data, temp_path)
                        print(f"[INFO] Temporary save at {processed_files} files: {len(self.samples)} samples")
                        
                        # Clear GPU cache
                        torch.cuda.empty_cache()

                print(f"[INFO] Precomputed {len(self.samples)} samples with DINO + Flow features.")
                
                # Save preprocessed results and text mapping
                save_data = {
                    'samples': self.samples,
                    'text_map': self.text_map,
                    'text_counter': self.text_counter
                }
                torch.save(save_data, features_path)
                print(f"[Rank {self.rank}] Preprocessing completed and saved.")
        
        # Distributed sync: wait for rank 0 to finish preprocessing
        if self.world_size > 1:
            torch.distributed.barrier()
        
        # All processes load the preprocessed results
        if os.path.exists(features_path):
            print(f"[Rank {self.rank}] Loading preprocessed features...")
            save_data = torch.load(features_path, map_location='cpu')
            self.samples = save_data['samples']
            self.text_map = save_data.get('text_map', {})
            self.text_counter = save_data.get('text_counter', 0)
            print(f"[Rank {self.rank}] Loaded {len(self.samples)} samples.")
        else:
            raise FileNotFoundError(f"Features file not found: {features_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['img_feat'], sample['flow_feat'], sample['action'], sample['text_id']