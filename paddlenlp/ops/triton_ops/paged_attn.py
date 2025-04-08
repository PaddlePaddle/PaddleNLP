# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/paged_attn.py

from typing import List, Optional, Tuple

import paddle
import numpy as np
from paddlenlp.ops.triton_ops.prefix_prefill import context_attention_fwd


class PagedAttention:
    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 80, 96, 112, 120, 128, 192, 256]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size * num_kv_heads * head_size)

    @staticmethod
    def split_kv_cache(
        kv_cache: paddle.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        x = 16 // kv_cache.element_size()
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.reshape([num_blocks, num_kv_heads, head_size // x, -1, x])

        value_cache = kv_cache[1]
        value_cache = value_cache.reshape([num_blocks, num_kv_heads, head_size, -1])
        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(
        key: paddle.Tensor,
        value: paddle.Tensor,
        key_cache: paddle.Tensor,
        value_cache: paddle.Tensor,
        slot_mapping: paddle.Tensor,
        kv_cache_dtype: str,
        k_scale: paddle.Tensor,
        v_scale: paddle.Tensor,
    ) -> None:
        from paddlenlp_ops import reshape_and_cache
        reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.flatten(),
            k_scale,
            v_scale,
            kv_cache_dtype
        )
    @staticmethod
    def forward_prefix(
        query,
        key,
        value,
        kv_cache_dtype: str,
        key_cache,
        value_cache,
        block_tables,
        query_start_loc,
        seq_lens_tensor,
        context_lens,
        max_query_len: int,
        alibi_slopes,
        sliding_window: Optional[int],
        k_scale,
        v_scale,
    ):
        output = paddle.empty_like(query)
        context_attention_fwd(
            query,
            key,
            value,
            output,
            kv_cache_dtype,
            key_cache,
            value_cache,
            block_tables,
            # query_start_loc is (batch_size + 1,)
            query_start_loc,
            seq_lens_tensor,
            context_lens,
            max_query_len,
            k_scale,
            v_scale,
            alibi_slopes,
            sliding_window,
        )
        return output

def compute_slot_mapping(
    seq_lens: List[int],
    query_lens: List[int],
    context_lens: List[int],
    block_tables: List[List[int]],
    block_size: int,
    sliding_window: int
) -> List[int]:
    PAD_SLOT_ID = -1
    _COMPUTE_SLOT_MAPPING_NUMPY_NUMEL=256
    slot_mapping = []
    
    for i in range(len(seq_lens)):
        seq_len = seq_lens[i]
        query_len = query_lens[i]
        context_len = context_lens[i]
        if i < len(block_tables):
            block_table = block_tables[i]
        else:
            slot_mapping.append(PAD_SLOT_ID)  
            continue
        
        is_profile_run = block_table is None
        if is_profile_run:
            slot_mapping.extend([PAD_SLOT_ID] * seq_len)
            continue
        
        is_prompt = query_len > 1
        
        start_idx = 0
        if is_prompt and sliding_window is not None:
            start_idx = max(0, query_len - sliding_window)
        
        padding_mask_len = max(0, start_idx - context_len)
        slot_mapping.extend([PAD_SLOT_ID] * padding_mask_len)
        
        range_start = max(start_idx, context_len)
        range_end = seq_len
        numel = range_end - range_start
        
        if numel <= 0:
            continue
        
        if numel < _COMPUTE_SLOT_MAPPING_NUMPY_NUMEL: 
            for j in range(range_start, range_end):
                block_idx = j // block_size
                if block_idx >= len(block_table):
                    slot_mapping.append(PAD_SLOT_ID)
                else:
                    block_number = block_table[block_idx]
                    slot = block_number * block_size + (j % block_size)
                    slot_mapping.append(slot)
        else:
            j_indices = np.arange(range_start, range_end)
            block_indices = j_indices // block_size
            valid_mask = block_indices < len(block_table)
            
            block_numbers = np.where(
                valid_mask,
                np.array(block_table)[block_indices],
                -1
            )
            offsets = j_indices % block_size
            slots = block_numbers * block_size + offsets
            slot_mapping.extend(slots.astype(np.int64).tolist())
    
    return slot_mapping