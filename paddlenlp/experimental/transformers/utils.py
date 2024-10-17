# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import os

import numpy as np
import paddle

from paddlenlp.transformers.model_utils import (
    dtype_guard,
    load_tp_checkpoint,
    no_init_weights,
)
from paddlenlp.transformers.utils import (
    ContextManagers,
    is_paddle_support_lazy_init,
    is_safetensors_available,
)


def infererence_model_from_pretrained(cls, pretrained_model_name_or_path, args, kwargs, return_numpy=True):
    r"""
    Instantiate a pretrained model configuration from a pre-trained model name or path.
    """
    config = kwargs.pop("config", None)
    cache_dir = kwargs.pop("cache_dir", None)
    dtype = kwargs.pop("dtype", None)
    if dtype is None:
        dtype = config.dtype
    subfolder = kwargs.pop("subfolder", None)
    if subfolder is None:
        subfolder = ""
    variant = kwargs.pop("variant", None)
    use_safetensors = kwargs.pop("use_safetensors", None if is_safetensors_available() else False)
    low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)

    init_contexts = []
    if low_cpu_mem_usage or config.quantization_config.is_weight_quantize():
        # Instantiate model.
        init_contexts.append(no_init_weights(_enable=True))
        if is_paddle_support_lazy_init():
            init_contexts.append(paddle.LazyGuard())
    if dtype:
        init_contexts.append(dtype_guard(dtype))

    # init the model
    with ContextManagers(init_contexts):
        model = cls(config)

    resolved_archive_file, _, _, _ = cls._resolve_model_file_path(
        pretrained_model_name_or_path,
        cache_dir=cache_dir,
        subfolder=subfolder,
        from_hf_hub=False,
        from_aistudio=False,
        config=config,
        convert_from_torch=False,
        use_safetensors=use_safetensors,
        variant=variant,
    )

    model_path = os.path.dirname(resolved_archive_file)
    state_dict = load_tp_checkpoint(model_path, cls, config, return_numpy=return_numpy)
    model.set_state_dict(state_dict)

    return model


class EmptyActScale:
    """
    For fake parameter
    """

    def __init__(
        self,
        key_map_dict=None,
        num_of_layers=None,
    ):
        self.key_map = key_map_dict
        self.scale = {}
        for scale_type, key_template in self.key_map.items():
            self.scale[scale_type] = np.full([num_of_layers], fill_value=0.1, dtype="float32")


class EmptyWeightScale:
    """
    For fake parameter
    """

    def __init__(
        self,
        key_map_dict,
        num_of_layers,
        num_heads,
        dim_head,
        ffn_hidden_size,
        num_key_value_heads=-1,
        mp_size=1,
        concat_qkv=False,
        concat_ffn1=False,
    ):
        self.key_map = key_map_dict
        self.scale = {}

        qkv_out_size = (
            3 * num_heads * dim_head if num_key_value_heads <= 0 else (num_heads + 2 * num_key_value_heads) * dim_head
        )

        for scale_type, key_template in self.key_map.items():
            if "qkv" in scale_type:
                n = qkv_out_size // mp_size
            elif "ffn1" in scale_type:
                n = ffn_hidden_size * 2 // mp_size
            else:
                n = num_heads * dim_head
            self.scale[scale_type] = np.full([num_of_layers, n], fill_value=0.1, dtype="float32")

        # concat qkv and ffn1
        if concat_qkv:
            self.scale["qkv_weight_scale"] = np.full(
                [num_of_layers, qkv_out_size // mp_size], fill_value=0.1, dtype="float32"
            )

        if concat_ffn1:
            self.scale["ffn1_weight_scale"] = np.full(
                [num_of_layers, ffn_hidden_size * 2 // mp_size], fill_value=0.1, dtype="float32"
            )


class EmptyCacheScale:
    """
    For fake parameter
    """

    def __init__(
        self,
        key_map_dict=None,
        num_of_layers=None,
        num_heads=None,
        dim_heads=None,
        is_channel_wise=False,
        mp_size=1,
        num_key_value_heads=-1,
    ):
        self.key_map = key_map_dict
        self.scale = {}

        num_heads = num_heads // mp_size
        num_key_value_heads = num_key_value_heads // mp_size
        kv_num_head = num_heads if num_key_value_heads <= 0 else num_key_value_heads
        for scale_type, key_template in self.key_map.items():
            if "cache_k" in scale_type:
                scale_type_out = "cache_k_out_scale"
            else:
                scale_type_out = "cache_v_out_scale"

            col_dim = kv_num_head * dim_heads if is_channel_wise else kv_num_head
            self.scale[scale_type] = np.full([num_of_layers, col_dim], fill_value=1.0)
            self.scale[scale_type_out] = np.full([num_of_layers, col_dim], fill_value=1.0)
