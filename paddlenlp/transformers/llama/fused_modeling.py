# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Paddle Llama model"""
from __future__ import annotations

import math
from functools import partial
from typing import Optional, Tuple, Type

import numpy as np
import paddle
import paddle.distributed.fleet.meta_parallel as mpu
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute
from paddle.incubate.nn import FusedMultiTransformer
from paddle.utils import try_import

from paddlenlp.transformers.conversion_utils import (
    StateDictNameMapping,
    init_name_mappings,
)

try:
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None

import warnings

from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from paddlenlp.transformers.model_utils import PretrainedModel, register_base_model

from .configuration import LlamaConfig

LLAMA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "__internal_testing__/tiny-random-llama",
    "facebook/llama-7b",
    "facebook/llama-13b",
]

__all__ = ["LlamaModel", "LlamaPretrainedModel", "LlamaForCausalLM", "LlamaPretrainingCriterion", "FusedLlamaModel"]


def get_triangle_upper_mask(x, mask=None):
    if mask is not None:
        return mask
    # [bsz, n_head, q_len, kv_seq_len]
    shape = x.shape
    #  [bsz, 1, q_len, kv_seq_len]
    shape[1] = 1
    mask = paddle.full(shape, -np.inf, dtype=x.dtype)
    mask.stop_gradient = True
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


def parallel_matmul(x: Tensor, y: Tensor, tensor_parallel_output=True):
    is_fleet_init = True
    tensor_parallel_degree = 1
    try:
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        tensor_parallel_degree = hcg.get_model_parallel_world_size()
    except:
        is_fleet_init = False

    if is_fleet_init and tensor_parallel_degree > 1 and y.is_distributed:
        # if not running under distributed.launch, it will raise AttributeError: 'Fleet' object has no attribute '_hcg'
        input_parallel = paddle.distributed.collective._c_identity(x, group=model_parallel_group)
        logits = paddle.matmul(input_parallel, y, transpose_y=False)

        if tensor_parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)

    else:
        logits = paddle.matmul(x, y, transpose_y=False)
        return logits


def finfo(dtype: paddle.dtype = None):
    if dtype is None:
        dtype = paddle.get_default_dtype()

    if dtype == paddle.bfloat16:
        # Numpy do not support `np.finfo(np.uint16)`, so try to construct a finfo object to fetch min value
        class BFloatFInfo:
            min = -3.3895313892515355e38

        return BFloatFInfo
    if dtype == paddle.float32:
        return np.finfo(np.float32)
    if dtype == paddle.float16:
        return np.finfo(np.float16)
    if dtype == paddle.float64:
        return np.finfo(np.float64)


def get_rotary_embedding(bsz, max_position_embeddings, base, head_dim, seq_length, offset): 
    inv_freq = 1.0 / (base ** (paddle.cast(paddle.arange(0, head_dim, 2), "float32") / head_dim))
    t = paddle.arange(max_position_embeddings, dtype=inv_freq.dtype)

    # shape: [S, D/2]
    freqs = paddle.einsum("i,j->ij", t, inv_freq)
    # shape: [S, D]
    emb = paddle.concat([freqs, freqs], axis=-1)

    # shape: [1, S, D]
    emb = paddle.unsqueeze(emb, 0)
    # shape: [1, S, 1, D]
    emb = paddle.unsqueeze(emb, 2)
    # shape: [B, S, 1, D]
    emb = paddle.repeat_interleave(emb, bsz, axis=0)

    cos_emb = paddle.cos(emb)
    sin_emb = paddle.sin(emb)
    stacked_rotary_emb = paddle.concat([cos_emb, sin_emb], axis=0)
    return stacked_rotary_emb[:, offset: seq_length+offset, :, :]

def fused_act_bias(
    x,
    bias=None,
    dequant_scales=None,
    shift=None,
    smooth=None,
    act_method='gelu',
    compute_dtype='default',
    rows=0,
    cols=0,
    quant_scale=-1,
    quant_round_type=0,
    quant_max_bound=0,
    quant_min_bound=0,
):
    return paddle._C_ops.fused_bias_act(
        x,
        bias,
        dequant_scales,
        shift,
        smooth,
        act_method,
        compute_dtype,
        rows,
        cols,
        quant_scale,
        quant_round_type,
        quant_max_bound,
        quant_min_bound,
    )

class LlamaMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        if config.tensor_parallel_degree > 1:
            self.gate_proj = mpu.ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size * 2,
                gather_output=False,
                has_bias=False,
            )
            self.down_proj = mpu.RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                input_is_parallel=True,
                has_bias=False,
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias_attr=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias_attr=False)

    def forward(self, x):
        gate_out = self.gate_proj(x)
        swiglu_act = fused_act_bias(gate_out, act_method="swiglu")
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class FusedLlamaRMSNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

    def forward(self, hidden_states, residual=None, bias=None):
        return paddle._C_ops.norm_helper(
            hidden_states,
            residual,
            bias,
            self.weight,
            None,
            self.variance_epsilon,
            1.0, # residual_alpha(not used.)
            "rmsnorm",
            begin_norm_axis=2,
        )[0:2]

def llama_fmha(
    qkv,
    seq_lens,
    padding_offset,
    pre_cache=None,
    mask=None,
    scale=None,
    causal=False,
):
    return paddle._C_ops.memory_efficient_attention_variable(
        qkv, 
        seq_lens, 
        padding_offset, 
        pre_cache, 
        mask, 
        scale, 
        causal,
    )
