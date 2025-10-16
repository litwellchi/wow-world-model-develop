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

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict1.diffusion.training.callbacks.iter_speed import IterSpeed
from cosmos_predict1.diffusion.training.callbacks.low_precision import LowPrecisionCallback
from cosmos_predict1.diffusion.training.datasets.dataset_video import Dataset
# from cosmos_predict1.diffusion.training.datasets.dataset_pd import DatasetPD as Dataset_pd
from cosmos_predict1.diffusion.training.datasets.dataset_pd import NEW_DatasetPD as Dataset_pd
from cosmos_predict1.diffusion.training.datasets.dataset_roborepa import DatasetRepa as Dataset_repa
from cosmos_predict1.diffusion.training.models.extend_model import FSDPExtendDiffusionModel
from cosmos_predict1.diffusion.training.models.model_peft import PEFTExtendDiffusionModel
from cosmos_predict1.diffusion.training.networks.general_dit_lvg import VideoExtendGeneralDIT
from cosmos_predict1.diffusion.training.utils.peft.lora_config import get_fa_ca_qv_lora_config
from cosmos_predict1.utils import log
from cosmos_predict1.utils.callback import ProgressBarCallback
from cosmos_predict1.utils.callbacks.grad_clip import GradClip
from cosmos_predict1.utils.lazy_config import PLACEHOLDER
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


cs = ConfigStore.instance()

n_length = 15 #15
num_frames = 8 * n_length + 1  # 121

# HDVILA example
example_video_dataset_hdvila = L(Dataset)(
    dataset_dir="datasets/hdvila",
    sequence_interval=1,
    num_frames=num_frames,
    video_size=(720, 1280),
    start_frame_interval=1,
)

dataloader_train_hdvila = L(DataLoader)(
    dataset=example_video_dataset_hdvila,
    sampler=L(get_sampler)(dataset=example_video_dataset_hdvila),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)

# Cosmos-NeMo-Assets example
example_video_dataset_cosmos_nemo_assets = L(Dataset)(
    dataset_dir="/mnt/pd/Data/cosmos_nemo_assets",
    sequence_interval=1,
    num_frames=num_frames,
    # video_size=(384, 384),
    video_size=(192, 192),
    # video_size=(720, 1280),
    start_frame_interval=1,
)

dataloader_train_cosmos_nemo_assets = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)

# Cosmos-NeMo-Assets examples with more affordable GPUs setup (4 GPUs or 40GB VRAM)
n_length_4gpu_80gb = 15
num_frames_4gpu_80gb = 8 * n_length_4gpu_80gb + 1  # 121
example_video_dataset_cosmos_nemo_assets_4gpu_80gb = L(Dataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    sequence_interval=1,
    num_frames=num_frames_4gpu_80gb,
    video_size=(384, 384),  # a low-res example for lower VRAM utilization without considering the content aspect ratio.
    start_frame_interval=1,
)

dataloader_train_cosmos_nemo_assets_4gpu_80gb = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_4gpu_80gb,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets_4gpu_80gb),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

n_length_8gpu_40gb = 3
num_frames_8gpu_40gb = 8 * n_length_8gpu_40gb + 1  # 25
example_video_dataset_cosmos_nemo_assets_8gpu_40gb = L(Dataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    sequence_interval=1,
    num_frames=num_frames_8gpu_40gb,
    video_size=(384, 384),  # a low-res example for lower VRAM utilization without considering aspect ratio.
    start_frame_interval=1,
)

dataloader_train_cosmos_nemo_assets_8gpu_40gb = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_8gpu_40gb,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets_8gpu_40gb),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

n_length_4gpu_40gb = 3
num_frames_4gpu_40gb = 8 * n_length_4gpu_40gb + 1  # 25
example_video_dataset_cosmos_nemo_assets_4gpu_40gb = L(Dataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    sequence_interval=1,
    num_frames=num_frames_4gpu_40gb,
    video_size=(192, 192),  # a low-res example for lower VRAM utilization without considering aspect ratio.
    start_frame_interval=1,
)

dataloader_train_cosmos_nemo_assets_4gpu_40gb = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_4gpu_40gb,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets_4gpu_40gb),
    batch_size=1,
    drop_last=True,
    num_workers=0,
    pin_memory=True,
)

# Cosmos-NeMo-Assets 480x848 example for lora
example_video_dataset_cosmos_nemo_assets_480_848 = L(Dataset)(
    dataset_dir="/mnt/pd/Data/cosmos_nemo_assets",
    sequence_interval=1,
    num_frames=num_frames,
    # video_size=(480, 848),
    video_size=(480, 480),
    start_frame_interval=1,
)

dataloader_train_cosmos_nemo_assets_480_848 = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_480_848,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets_480_848),
    batch_size=2,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)

dataloader_val_cosmos_nemo_assets_480_848 = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_480_848,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets_480_848),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)

video2world_7b_example_hdvila = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "fsdp"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_example_hdvila",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=200,
            broadcast_via_filesystem=False,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
        ),
        trainer=dict(
            max_iter=2000,
            distributed_parallelism="fsdp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=True,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
        ),
        model=dict(
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                88,  # Latent height dim
                160,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=True,
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=32,
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(pixel_chunk_duration=num_frames),
        ),
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader_train_hdvila,
    )
)


video2world_7b_example_cosmos_nemo_assets = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "fsdp"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_example_cosmos_nemo_assets",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=200,
            broadcast_via_filesystem=False,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
        ),
        trainer=dict(
            max_iter=2000,
            distributed_parallelism="fsdp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=True,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
        ),
        model=dict(
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                88,  # Latent height dim
                160,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=True,
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=32,
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(pixel_chunk_duration=num_frames),
        ),
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader_train_cosmos_nemo_assets,
    )
)

video2world_7b_example_cosmos_nemo_assets_4gpu_80gb = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "fsdp"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_example_cosmos_nemo_assets_4gpu_80gb",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=200,
            broadcast_via_filesystem=False,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
        ),
        trainer=dict(
            max_iter=2000,
            distributed_parallelism="fsdp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=True,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
        ),
        model=dict(
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                48,  # Latent height dim
                48,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=True,
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=32,
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(
                pixel_chunk_duration=num_frames_4gpu_80gb,
                spatial_resolution="384",
            ),
        ),
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader_train_cosmos_nemo_assets_4gpu_80gb,
    )
)

video2world_7b_example_cosmos_nemo_assets_8gpu_40gb = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "fsdp"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_example_cosmos_nemo_assets_8gpu_40gb",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=200,
            broadcast_via_filesystem=False,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
            async_saving=False,  # set to False to save memory
        ),
        trainer=dict(
            max_iter=2000,
            distributed_parallelism="fsdp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=True,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
        ),
        model=dict(
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                48,  # Latent height dim
                48,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=False,  # turn off to save memory
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=32,
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(
                pixel_chunk_duration=num_frames_8gpu_40gb,
                spatial_resolution="384",
            ),
        ),
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader_train_cosmos_nemo_assets_8gpu_40gb,
    )
)

video2world_7b_example_cosmos_nemo_assets_4gpu_40gb = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "fsdp"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_example_cosmos_nemo_assets_4gpu_40gb",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=200,
            broadcast_via_filesystem=False,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
            async_saving=False,  # set to False to save memory
        ),
        trainer=dict(
            max_iter=2000,
            distributed_parallelism="fsdp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=True,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
        ),
        model=dict(
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                24,  # Latent height dim
                24,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=False,  # turn off to save memory
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=32,
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(
                pixel_chunk_duration=num_frames_4gpu_40gb,
                spatial_resolution="192",
            ),
        ),
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader_train_cosmos_nemo_assets_4gpu_40gb,
    )
)

video2world_7b_lora_example_cosmos_nemo_assets = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "peft"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_lora_example_cosmos_nemo_assets",
        ),
        optimizer=dict(
            lr=1e-4,
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=1000,
            broadcast_via_filesystem=True,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
            async_saving=False,  # set to False to save memory
        ),
        trainer=dict(
            max_iter=5000,
            distributed_parallelism="ddp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=False,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=2, #4
        ),
        model=dict(
            peft_control=get_fa_ca_qv_lora_config(first_nblocks=28, rank=8, scale=1),
            latent_shape=[
                16,
                16,
                88,
                160,
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=False,  # turn off to save memory
            ),
            fsdp_enabled=False,
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(pixel_chunk_duration=num_frames),
        ),
        model_obj=L(PEFTExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        scheduler=dict(
            warm_up_steps=[0],
        ),
        dataloader_train=dataloader_train_cosmos_nemo_assets_480_848,
        dataloader_val=dataloader_val_cosmos_nemo_assets_480_848,
    )
)

example_video_dataset_agibot_droid_480_480 = L(Dataset_pd)(
    dataset_path="/mnt/pd_data/0518_200k.jsonl",
    t5_dir="/mnt/pd/Data/t5_0517",
    sequence_interval=1,
    num_frames=num_frames,
    # video_size=(480, 848),
    video_size=(480, 480),
    start_frame_interval=1,
)

dataloader_train_agibot_droid_480_480 = L(DataLoader)(
    dataset=example_video_dataset_agibot_droid_480_480,
    sampler=L(get_sampler)(dataset=example_video_dataset_agibot_droid_480_480),
    batch_size=4,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)

dataloader_val_agibot_droid_480_480 = L(DataLoader)(
    dataset=example_video_dataset_agibot_droid_480_480,
    sampler=L(get_sampler)(dataset=example_video_dataset_agibot_droid_480_480),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)

# agibot droid 200k data 204136
example_video_dataset_agibot_droid_720_720 = L(Dataset_pd)(
    dataset_path="/mnt/pd_data/0613_200k_longtext.jsonl",
    t5_dir="/mnt/pd/Data/t5_0613",
    sequence_interval=1,
    num_frames=num_frames,
    video_size=(720, 1024),
    start_frame_interval=1,
)

# robomind agibot droid 600k data 
example_video_dataset_robomind_agibot_droid_train = L(Dataset_pd)(
    dataset_path="/mnt/world_foundational_model/pd_data/0627_agibot_droid_robomind_600k_longtext.jsonl",
    t5_dir="/mnt/world_foundational_model/pd/Data/t5_0613",
    sequence_interval=1,
    num_frames=num_frames,
    video_size=(720, 1024),
    start_frame_interval=1,
)

example_video_dataset_robomind_agibot_droid_val = L(Dataset_pd)(
    dataset_path="/mnt/world_foundational_model/pd_data/0614_50_longtext.jsonl",
    t5_dir="/mnt/world_foundational_model/pd/Data/t5_0613",
    sequence_interval=1,
    num_frames=num_frames,
    video_size=(720, 1024),
    start_frame_interval=1,
)

dataloader_robomind_agibot_droid_train = L(DataLoader)(
    dataset=example_video_dataset_robomind_agibot_droid_train,
    sampler=L(get_sampler)(dataset=example_video_dataset_robomind_agibot_droid_train),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=16,
)

dataloader_robomind_agibot_droid_val = L(DataLoader)(
    dataset=example_video_dataset_robomind_agibot_droid_val,
    sampler=L(get_sampler)(dataset=example_video_dataset_robomind_agibot_droid_val),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)

# dataset_tienkung

# tienkung_train = L(Dataset_pd)(
#     dataset_path="/mnt/world_foundational_model/pd_data/TienKung/tienkung_0803.jsonl",
#     t5_dir="/mnt/world_foundational_model/pd_data/TienKung/t5_embeds",
#     sequence_interval=1,
#     num_frames=num_frames,
#     video_size=(720, 1024),
#     start_frame_interval=1,
# )

# tienkung_val = L(Dataset_pd)(
#     dataset_path="/mnt/world_foundational_model/pd_data/TienKung/tienkung_0803.jsonl",
#     t5_dir="/mnt/world_foundational_model/pd_data/TienKung/t5_embeds",
#     sequence_interval=1,
#     num_frames=num_frames,
#     video_size=(720, 1024),
#     start_frame_interval=1,
# )

tienkung_train = L(Dataset_pd)(
    # dataset_path="/mnt/world_foundational_model/pd/Data/Test/Tienkung/test.jsonl",
    # t5_dir="/mnt/world_foundational_model/pd_data/TienKung/t5_embeds",
    dataset_path="/mnt/world_foundational_model/pd/Data/Test/Tienkung_new/test_longtext.jsonl",
    t5_dir="/mnt/world_foundational_model/pd_data/TianYi/t5_embeds",
    sequence_interval=1,
    num_frames=num_frames,
    video_size=(720, 1024),
    start_frame_interval=1,
)

tienkung_val = L(Dataset_pd)(
    # dataset_path="/mnt/world_foundational_model/pd/Data/Test/Tienkung/test.jsonl",
    # t5_dir="/mnt/world_foundational_model/pd_data/TienKung/t5_embeds",
    dataset_path="/mnt/world_foundational_model/pd/Data/Test/Tienkung_new/test_longtext.jsonl",
    t5_dir="/mnt/world_foundational_model/pd_data/TianYi/t5_embeds",
    sequence_interval=1,
    num_frames=num_frames,
    video_size=(720, 1024),
    start_frame_interval=1,
)

dataloader_robomind_tienkung_train = L(DataLoader)(
    dataset=tienkung_train,
    sampler=L(get_sampler)(dataset=tienkung_train),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=16,
)

dataloader_robomind_tienkung_val = L(DataLoader)(
    dataset=tienkung_val,
    sampler=L(get_sampler)(dataset=tienkung_val),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)

real_30k_train = L(Dataset_pd)(
    # dataset_path="/mnt/world_foundational_model/pd/Data/Test/Tienkung/test.jsonl",
    # t5_dir="/mnt/world_foundational_model/pd_data/TienKung/t5_embeds",
    dataset_path="/mnt/world_foundational_model/pd_data/0904_real_30k_longtext.jsonl",
    t5_dir="/mnt/world_foundational_model/pd_data/0904_real_30k_t5/",
    sequence_interval=1,
    num_frames=num_frames,
    video_size=(720, 1024),
    start_frame_interval=1,
)

real_30k_val = L(Dataset_pd)(
    # dataset_path="/mnt/world_foundational_model/pd/Data/Test/Tienkung/test.jsonl",
    # t5_dir="/mnt/world_foundational_model/pd_data/TienKung/t5_embeds",
    dataset_path="/mnt/world_foundational_model/pd_data/0904_real_30k_longtext.jsonl",
    t5_dir="/mnt/world_foundational_model/pd_data/0904_real_30k_t5/",
    sequence_interval=1,
    num_frames=num_frames,
    video_size=(720, 1024),
    start_frame_interval=1,
)

dataloader_real_30k_train = L(DataLoader)(
    dataset=real_30k_train,
    sampler=L(get_sampler)(dataset=real_30k_train),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=16,
)

dataloader_real_30k_val = L(DataLoader)(
    dataset=real_30k_val,
    sampler=L(get_sampler)(dataset=real_30k_val),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)


dataloader_train_agibot_droid_720_720 = L(DataLoader)(
    dataset=example_video_dataset_agibot_droid_720_720,
    sampler=L(get_sampler)(dataset=example_video_dataset_agibot_droid_720_720),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=16,
)

example_video_dataset_agibot_droid_720_720_val = L(Dataset_pd)(
    dataset_path="/mnt/pd_data/0614_50_longtext.jsonl",
    t5_dir="/mnt/pd/Data/t5_0613",
    sequence_interval=1,
    num_frames=num_frames,
    video_size=(720, 1024),
    start_frame_interval=1,
)

dataloader_val_agibot_droid_720_720 = L(DataLoader)(
    dataset=example_video_dataset_agibot_droid_720_720_val,
    sampler=L(get_sampler)(dataset=example_video_dataset_agibot_droid_720_720_val),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)

example_video_dataset_agibot_droid_384_384 = L(Dataset_repa)(
    dataset_path="/mnt/pd_data/0613_200k_longtext.jsonl",
    t5_dir="/mnt/pd/Data/t5_0613",
    sequence_interval=1,
    num_frames=num_frames,
    video_size=(384, 384),
    start_frame_interval=1,
)

dataloader_train_agibot_droid_384_384 = L(DataLoader)(
    dataset=example_video_dataset_agibot_droid_384_384,
    sampler=L(get_sampler)(dataset=example_video_dataset_agibot_droid_384_384),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=16,
)

example_video_dataset_agibot_droid_384_384_val = L(Dataset_repa)(
    dataset_path="/mnt/pd_data/0614_50_longtext.jsonl",
    t5_dir="/mnt/pd/Data/t5_0613",
    sequence_interval=1,
    num_frames=num_frames,
    video_size=(384, 384),
    start_frame_interval=1,
)

dataloader_val_agibot_droid_384_384 = L(DataLoader)(
    dataset=example_video_dataset_agibot_droid_384_384_val,
    sampler=L(get_sampler)(dataset=example_video_dataset_agibot_droid_384_384_val),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)

video2world_7b_lora_agibot_droid = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "peft"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_lora_agibot_droid",
        ),
        optimizer=dict(
            lr=1e-4,
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=1000,
            broadcast_via_filesystem=True,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
            async_saving=False,  # set to False to save memory
        ),
        trainer=dict(
            max_iter=20000,
            distributed_parallelism="ddp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=False,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=True, #False
            tensor_model_parallel_size=2,
            context_parallel_size=1, #4
        ),
        model=dict(
            peft_control=get_fa_ca_qv_lora_config(first_nblocks=28, rank=8, scale=1),
            latent_shape=[
                16,
                16,
                88,
                160,
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=False,  # turn off to save memory
            ),
            fsdp_enabled=False,
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(pixel_chunk_duration=num_frames),
        ),
        model_obj=L(PEFTExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        scheduler=dict(
            warm_up_steps=[0],
        ),
        dataloader_train=dataloader_train_agibot_droid_480_480,
        dataloader_val=dataloader_val_agibot_droid_480_480,
    )
)

video2world_7b_lora_agibot_droid_ms = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "peft"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_lora_agibot_droid_ms",
        ),
        optimizer=dict(
            lr=1e-4,
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=2000,
            broadcast_via_filesystem=True,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
            async_saving=False,  # set to False to save memory
        ),
        trainer=dict(
            max_iter=300000,
            distributed_parallelism="ddp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=False,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=True,
            tensor_model_parallel_size=2,
            context_parallel_size=4, #4
        ),
        model=dict(
            peft_control=get_fa_ca_qv_lora_config(first_nblocks=28, rank=8, scale=1),
            latent_shape=[
                16,
                16,
                88,
                160,
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=False,  # turn off to save memory
            ),
            fsdp_enabled=False,
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(pixel_chunk_duration=num_frames),
        ),
        model_obj=L(PEFTExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        scheduler=dict(
            warm_up_steps=[0],
        ),
        dataloader_train=dataloader_train_agibot_droid_480_480,
        dataloader_val=dataloader_val_agibot_droid_480_480,
    )
)


video2world_7b_lora_agibot_droid_ms_resume = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "peft"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_lora_agibot_droid_ms",
        ),
        optimizer=dict(
            lr=1e-4,
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=2000,
            broadcast_via_filesystem=True,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=True,
            strict_resume=False,
            keys_not_to_resume=[],
            async_saving=False,  # set to False to save memory
        ),
        trainer=dict(
            max_iter=100000,
            distributed_parallelism="ddp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=False,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=True,
            tensor_model_parallel_size=2,
            context_parallel_size=4, #4
        ),
        model=dict(
            peft_control=get_fa_ca_qv_lora_config(first_nblocks=28, rank=8, scale=1),
            latent_shape=[
                16,
                16,
                88,
                160,
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=False,  # turn off to save memory
            ),
            fsdp_enabled=False,
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(pixel_chunk_duration=num_frames),
        ),
        model_obj=L(PEFTExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        scheduler=dict(
            warm_up_steps=[0],
        ),
        dataloader_train=dataloader_train_agibot_droid_480_480,
        dataloader_val=dataloader_val_agibot_droid_480_480,
    )
)

video_mini = L(Dataset_pd)(
    dataset_path="/mnt/pd/Data/Test/Cosmos/test_0519.jsonl",
    t5_dir="/mnt/pd/Data/t5_0517",
    sequence_interval=1,
    num_frames=num_frames,
    # video_size=(480, 848),
    video_size=(480, 480),
    start_frame_interval=1,
)

dataloader_train_mini = L(DataLoader)(
    dataset=video_mini,
    sampler=L(get_sampler)(dataset=video_mini),
    batch_size=2,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)

dataloader_val_mini = L(DataLoader)(
    dataset=video_mini,
    sampler=L(get_sampler)(dataset=video_mini),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)

video_400 = L(Dataset_pd)(
    dataset_path="/mnt/pd/Data/Train_mini/train_mini.jsonl",
    t5_dir="/mnt/pd/Data/t5_0517",
    sequence_interval=1,
    num_frames=num_frames,
    # video_size=(480, 848),
    video_size=(480, 480),
    start_frame_interval=1,
)

dataloader_train_400 = L(DataLoader)(
    dataset=video_400,
    sampler=L(get_sampler)(dataset=video_400),
    batch_size=2,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)

dataloader_val_400 = L(DataLoader)(
    dataset=video_400,
    sampler=L(get_sampler)(dataset=video_400),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)


video2world_7b_sft_400 = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "fsdp"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_sft_400",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=4000,
            broadcast_via_filesystem=False,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
            async_saving=False,  # set to False to save memory
        ),
        trainer=dict(
            max_iter=50000,
            distributed_parallelism="fsdp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=True,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
        ),
        model=dict(
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                48,  # Latent height dim
                48,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=False,  # turn off to save memory
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=8,
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(
                pixel_chunk_duration=num_frames,
                spatial_resolution="384",
            ),
        ),
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader_train_400,
        dataloader_val=dataloader_val_400,
    )
)

video2world_7b_lora_mini = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "peft"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_lora_mini",
        ),
        optimizer=dict(
            lr=1e-4,
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=1000,
            broadcast_via_filesystem=True,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
            async_saving=False,
        ),
        trainer=dict(
            max_iter=50000,
            distributed_parallelism="ddp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=False,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=4, #4
        ),
        model=dict(
            peft_control=get_fa_ca_qv_lora_config(first_nblocks=28, rank=8, scale=1),
            latent_shape=[
                16,
                16,
                88,
                160,
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=True,  # turn off to save memory
            ),
            fsdp_enabled=False,
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(pixel_chunk_duration=num_frames),
        ),
        model_obj=L(PEFTExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        scheduler=dict(
            warm_up_steps=[0],
        ),
        dataloader_train=dataloader_train_mini,
        dataloader_val=dataloader_val_mini,
    )
)

video2world_7b_sft_200k_longtext_720p = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "fsdp"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_sft_200k_longtext_720p",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=500,
            broadcast_via_filesystem=False,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
            async_saving=False,  # set to False to save memory
        ),
        trainer=dict(
            max_iter=60000,
            distributed_parallelism="fsdp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=True,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
        ),
        model=dict(
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                90,  # Latent height dim
                128,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=False,  # turn off to save memory
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=8, # 32
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(
                pixel_chunk_duration=num_frames,
                spatial_resolution="720",
            ),
        ),
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader_robomind_agibot_droid_train,
        dataloader_val=dataloader_robomind_agibot_droid_val,
    )
)

video2world_7b_sft_tienkung = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "fsdp"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="video2world_tienkung",
            name="video2world_7b_sft_tienkung",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=100, #500
            broadcast_via_filesystem=False,
            load_path="/mnt/world_foundational_model/pd/Checkpoints/posttraining/diffusion_video2world/video2world_7b_sft_200k_longtext_720p/checkpoints/iter_000033000_reg_model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
            async_saving=False,  # set to False to save memory
        ),
        trainer=dict(
            max_iter=60000,
            distributed_parallelism="fsdp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=True,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
        ),
        model=dict(
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                90,  # Latent height dim
                128,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=False,  # turn off to save memory
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=8, # 32
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(
                pixel_chunk_duration=num_frames,
                spatial_resolution="720",
            ),
        ),
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader_robomind_tienkung_train,
        dataloader_val=dataloader_robomind_tienkung_val,
    )
)


video2world_7b_sft_30k = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "fsdp"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="video2world_30k",
            name="video2world_7b_sft_30k",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=100, #500
            broadcast_via_filesystem=False,
            load_path="/mnt/world_foundational_model/pd/Checkpoints/posttraining/diffusion_video2world/video2world_7b_sft_200k_longtext_720p/checkpoints/iter_000033000_reg_model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
            async_saving=False,  # set to False to save memory
        ),
        trainer=dict(
            max_iter=60000,
            distributed_parallelism="fsdp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=True,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
        ),
        model=dict(
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                90,  # Latent height dim
                128,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=False,  # turn off to save memory
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=8, # 32
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(
                pixel_chunk_duration=num_frames,
                spatial_resolution="720",
            ),
        ),
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader_real_30k_train,
        dataloader_val=dataloader_real_30k_val,
    )
)

video2world_7b_lora_tienkung = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "peft"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="video2world_7b_lora_tienkung",
            name="video2world_7b_lora_tienkung",
        ),
        optimizer=dict(
            lr=1e-4, 
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=300,
            broadcast_via_filesystem=True,
            load_path="/mnt/world_foundational_model/pd/Checkpoints/posttraining/diffusion_video2world/video2world_7b_sft_200k_longtext_720p/checkpoints/iter_000033000_reg_model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
            async_saving=False,  # set to False to save memory
        ),
        trainer=dict(
            max_iter=300000,
            distributed_parallelism="ddp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=False,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=True,
            tensor_model_parallel_size=2,
            context_parallel_size=2, #4
        ),
        model=dict(
            peft_control=get_fa_ca_qv_lora_config(first_nblocks=28, rank=8, scale=1),
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                90,  # Latent height dim
                128,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=True,  # turn off to save memory
            ),
            fsdp_enabled=False,
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(pixel_chunk_duration=num_frames),
        ),
        model_obj=L(PEFTExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        scheduler=dict(
            warm_up_steps=[20],
        ),
        dataloader_train=dataloader_robomind_tienkung_train,
        dataloader_val=dataloader_robomind_tienkung_val,
    )
)

video2world_7b_repa = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "fsdp"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_repa",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=500,
            broadcast_via_filesystem=False,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
            async_saving=False,  # set to False to save memory
        ),
        trainer=dict(
            max_iter=50000,
            distributed_parallelism="fsdp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=True,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
        ),
        model=dict(
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                48,  # Latent height dim
                48,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=False,  # turn off to save memory
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=4, # 32
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(
                pixel_chunk_duration=num_frames,
                spatial_resolution="720",
            ),
        ),
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader_train_agibot_droid_384_384,
        dataloader_val=dataloader_val_agibot_droid_384_384,
    )
)


def register_experiments(cs):
    # Register the experiments
    for _item in [
        video2world_7b_example_hdvila,
        video2world_7b_example_cosmos_nemo_assets,
        video2world_7b_example_cosmos_nemo_assets_4gpu_80gb,
        video2world_7b_example_cosmos_nemo_assets_8gpu_40gb,
        video2world_7b_example_cosmos_nemo_assets_4gpu_40gb,
        video2world_7b_lora_example_cosmos_nemo_assets,
        video2world_7b_lora_agibot_droid,
        video2world_7b_lora_agibot_droid_ms,
        video2world_7b_lora_mini,
        video2world_7b_sft_400,
        video2world_7b_sft_200k_longtext_720p,
        video2world_7b_sft_tienkung,
        video2world_7b_sft_30k,
        video2world_7b_lora_tienkung,
        video2world_7b_repa
    ]:
        experiment_name = _item["job"]["name"]
        log.info(f"Registering experiment: {experiment_name}")
        cs.store(
            group="experiment",
            package="_global_",
            name=experiment_name,
            node=_item,
        )
