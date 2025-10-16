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

from typing import Optional, Tuple

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from typing import Any, Dict, List, Optional, Set

from cosmos_predict1.autoregressive.networks.transformer import Transformer

import random

def sample_top_p(logits, temperature, top_p, return_probs: bool = False):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        logits (torch.Tensor): Logits of the probability distribution.
        temperature (float): Temperature for sampling.
        top_p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
    # Sort the probabilities in descending order and get their indices.
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # Compute the cumulative sum of the sorted probabilities.
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # Create a mask where the cumulative probability exceeds the threshold p.
    mask = probs_sum - probs_sort > top_p
    # Set the probabilities that exceed the threshold to 0.
    probs_sort[mask] = 0.0
    # Renormalize the remaining probabilities so they sum to 1.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # Sample from the renormalized probability distribution.
    # next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = multinomial_sample_one_no_sync(probs_sort, dtype=torch.int64)
    # Gather the indices of the sampled tokens.
    next_token = torch.gather(probs_idx, -1, next_token)
    if return_probs:
        # Initialize a tensor for unsorted probabilities
        probs_unsorted = torch.zeros_like(probs_sort)
        # Scatter the sorted probabilities back to their original order
        probs_unsorted.scatter_(-1, probs_idx, probs_sort)
    else:
        probs_unsorted = None
    return next_token, probs_unsorted


def multinomial_sample_one_no_sync(probs_sort, dtype=torch.int):
    """
    Multinomial sampling without a cuda synchronization.
    Source: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
    """
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=dtype)


def logits_to_probs(
    logits,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample_top_k(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    """
    Sample from the logits using top-k sampling.
    Source: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
    """
    # logits: [batch_size, seq_len, vocab_size]
    if temperature == 0.0:
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        probs = None
    else:
        probs = logits_to_probs(logits[:, -1, :], temperature, top_k)
        idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(
    model: Transformer,
    input_pos: torch.Tensor,
    tokens: torch.Tensor = None,
    token_embeddings: torch.Tensor = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    **kwargs,
) -> torch.Tensor:
    logits = model(tokens=tokens, token_embeddings=token_embeddings, input_pos=input_pos, **kwargs)
    # Only top-p or top-k can be provided
    assert (
        top_p is None or top_k is None
    ), "Only one of top-p or top-k can be provided, got top-p={top_p} and top-k={top_k}"
    if top_p is not None:
        return sample_top_p(logits, temperature=temperature, top_p=top_p)[0]
    else:
        return sample_top_k(logits, temperature=temperature, top_k=top_k)[0]


def decode_one_token(
    model: Transformer,
    tokens: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decode a single token from the autoregressive model.
    """
    logits = model(tokens=tokens, input_pos=input_pos, **kwargs)
    if top_p is not None:
        return sample_top_p(logits, temperature=temperature, top_p=top_p)
    else:
        return sample_top_k(logits, temperature=temperature, top_k=top_k)

def generate_mask(tensor, mask_ratio, mask_type='uniform', prompt_len=0):
    """
    生成掩码矩阵的函数，支持三种掩码策略
    
    参数:
        tensor (torch.Tensor): 输入张量，形状为 [B, L]
        mask_ratio (float): 掩码比例 (0-1)
        mask_type (str): 掩码类型 ('uniform', 'random', 'similar')
        prompt_len (int): 前 prompt_len 个位置不掩码且不计入比例计算
        
    返回:
        torch.Tensor: 掩码矩阵，形状为 [B, L]，1 表示被掩码的位置
    """
    B, L = len(tensor), len(tensor[0])
    
    if mask_ratio >= 1.0:
        mask = torch.ones((B, L), dtype=torch.bool)
        return mask
    
    mask = torch.zeros((B, L), dtype=torch.bool)
    # 可掩码区域长度（排除 prompt 部分）
    maskable_len = L - prompt_len
    if maskable_len <= 0 or mask_ratio <= 0:
        return mask
    
    # 计算实际需要掩码的数量
    mask_num = int(maskable_len * mask_ratio)
    if mask_num == 0:
        return mask
    
    if mask_type not in ['uniform', 'random', 'similar', 'continous']:
        raise ValueError("mask_type 必须是 'uniform', 'random', 'similar', 'continous' 之一")
    
    mask = torch.zeros(L, dtype=torch.bool)
    maskable_indices_abs = torch.arange(prompt_len, L)
    maskable_indices_rel = torch.arange(0, maskable_len)

    i=0
    if mask_type == 'uniform':
        if mask_ratio > 0:
            N = max(1, int(round(1.0 / mask_ratio)))
            indices_to_mask_uniform = torch.arange(prompt_len, L, step=N)
            if indices_to_mask_uniform.numel() > 0: 
                mask[indices_to_mask_uniform] = True
    elif mask_type == 'random':
        perm = torch.randperm(maskable_len)
        indices_to_mask_rel = perm[:mask_num]
        actual_indices_to_mask = maskable_indices_abs[indices_to_mask_rel]
        mask[actual_indices_to_mask] = True
    elif mask_type == "continuous":
        current_num_to_mask = min(mask_num, maskable_len)
        if current_num_to_mask > 0:
            max_start_offset = maskable_len - current_num_to_mask
            if max_start_offset < 0 : 
                start_offset = 0
                current_num_to_mask = maskable_len 
            else:
                start_offset = torch.randint(0, max_start_offset + 1, (1,)).item()

            start_abs = prompt_len + start_offset
            end_abs = start_abs + current_num_to_mask
            mask[start_abs:end_abs] = True
    else:
        active_similar_diff_levels = [0, 1, 2, 3, 4, 5]
        potential_indices_by_priority = [[] for _ in active_similar_diff_levels]

        for j in range(prompt_len, L - 1): 
            val1 = tensor[i][j]
            val2 = tensor[i][j+1]
            diff = abs(val1 - val2)

            for k_idx, diff_thresh in enumerate(active_similar_diff_levels):
                if diff == diff_thresh: 
                    potential_indices_by_priority[k_idx].append(j + 1) 
                    break 


        final_indices_to_mask_this_sample = set() 

        for k_idx in range(len(active_similar_diff_levels)): 
            candidates_at_this_level = potential_indices_by_priority[k_idx]
            random.shuffle(candidates_at_this_level) 

            for candidate_idx in candidates_at_this_level:
                if len(final_indices_to_mask_this_sample) < mask_num:
                    final_indices_to_mask_this_sample.add(candidate_idx)
                else:
                    break 
            
            if len(final_indices_to_mask_this_sample) == mask_num:
                break 

        if len(final_indices_to_mask_this_sample) < mask_num:
            num_needed_randomly = mask_num - len(final_indices_to_mask_this_sample)
            
            all_maskable_indices_list_abs = list(range(prompt_len, L))
            available_for_random_fill = [
                idx for idx in all_maskable_indices_list_abs 
                if idx not in final_indices_to_mask_this_sample
            ]
            
            num_to_pick_randomly = min(num_needed_randomly, len(available_for_random_fill))
            
            if num_to_pick_randomly > 0:
                randomly_chosen_for_fill = random.sample(available_for_random_fill, num_to_pick_randomly)
                for idx in randomly_chosen_for_fill:
                    final_indices_to_mask_this_sample.add(idx)

        if final_indices_to_mask_this_sample:
            mask[list(final_indices_to_mask_this_sample)] = True
    final_mask_batched = mask.unsqueeze(0).expand(B, L)
    
    return final_mask_batched

def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    stop_tokens: torch.Tensor = None,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    return_probs: bool = False,
    decode_one_token_function=decode_one_token,
    gt_tokens: List[List[int]] | torch.Tensor = None,
    mask_ratio: float = 0.0,
    mask_type: str = 'uniform',
    **kwargs,
):
    """
    Decode n tokens from the autoregressive model.
    Adapted from https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
    """
    new_tokens, new_probs = [], []
    batch_size = cur_token.shape[0]
    assert (
        top_p is None or top_k is None
    ), "Only one of top-p or top-k can be provided, got top-p={top_p} and top-k={top_k}"
    if stop_tokens is not None:
        # Indicator for whether the EOS token (stop token) has been reached for each sample in the batch
        eos_reached = torch.tensor([False] * batch_size, device="cuda")
    mask = generate_mask(gt_tokens, mask_ratio=mask_ratio, mask_type=mask_type, prompt_len=int(input_pos[0]))
    for t in range(num_new_tokens):
        with sdpa_kernel([SDPBackend.MATH]):  # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token_function(
                model,
                tokens=cur_token,
                input_pos=input_pos,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                **kwargs,
            )
            input_pos += 1
            
            if mask[0, int(input_pos[0])] == 1:
                # print(next_token)
                tmp = torch.tensor([[gt_tokens[0][int(input_pos[0])]]], device=next_token.device)
                next_token = tmp
                
            if stop_tokens is not None and len(stop_tokens) > 0:
                eos_reached = eos_reached | (torch.isin(next_token, stop_tokens))
                if eos_reached.all():
                    break
            new_tokens.append(next_token.clone())
            if return_probs:
                new_probs.append(next_prob.clone())
            cur_token = next_token.clone()

    if return_probs:
        return new_tokens, new_probs
    else:
        return new_tokens
