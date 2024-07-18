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

import json
import os
from functools import partial

import numpy as np
import paddle
from tqdm import tqdm

from paddlenlp.transformers import AutoConfig
from paddlenlp.transformers.model_utils import _add_variant, load_state_dict
from paddlenlp.transformers.utils import paddlenlp_load
from paddlenlp.utils.env import (
    PADDLE_WEIGHTS_INDEX_NAME,
    SAFE_MASTER_WEIGHTS_INDEX_NAME,
    SAFE_PEFT_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
)

try:
    from paddlenlp.utils.safetensors import fast_load_file as safe_load_file
    from paddlenlp.utils.safetensors import fast_safe_open as safe_open
except:
    from safetensors import safe_open
    from safetensors.numpy import load_file as safe_load_file


def load_sharded_checkpoint(folder, variant=None, return_numpy=False):
    """

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        folder (`str` or `os.PathLike`): A path to a folder containing the sharded checkpoint.
        variant (`str`): The model variant.
        return_numpy (`bool`): Whether to return numpy array instead of paddle tensor.

    """
    # Load the index
    pdparams_file = os.path.join(folder, _add_variant("model_state.pdparams", variant))
    lora_pdparams_file = os.path.join(folder, _add_variant("lora_model_state.pdparams", variant))
    safetensors_file = os.path.join(folder, _add_variant("model.safetensors", variant))
    if os.path.isfile(pdparams_file):
        return paddle.load(pdparams_file, return_numpy=return_numpy)
    if os.path.isfile(lora_pdparams_file):
        return paddle.load(lora_pdparams_file, return_numpy=return_numpy)
    if os.path.isfile(safetensors_file):
        state_dict = safe_load_file(safetensors_file)
        if not return_numpy:
            for key in list(state_dict.keys()):
                if isinstance(state_dict[key], np.ndarray):
                    state_dict[key] = paddle.Tensor(state_dict.pop(key), zero_copy=True)
        return state_dict

    index_file = os.path.join(folder, _add_variant(PADDLE_WEIGHTS_INDEX_NAME, variant))
    safe_index_file = os.path.join(folder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
    safe_master_file = os.path.join(folder, _add_variant(SAFE_MASTER_WEIGHTS_INDEX_NAME, variant))
    safe_peft_file = os.path.join(folder, _add_variant(SAFE_PEFT_WEIGHTS_INDEX_NAME, variant))

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)
    safe_master_present = os.path.isfile(safe_master_file)
    safe_peft_present = os.path.isfile(safe_peft_file)

    load_safe = False
    load_index = None
    if safe_index_present:
        load_safe = True  # load safe due to preference
        load_index = safe_index_file
    elif safe_master_present:
        load_safe = True
        load_index = safe_master_file
    elif index_present:
        load_index = index_file
    elif safe_peft_present:
        load_safe = True
        load_index = safe_peft_file
    else:
        raise ValueError(f"Could not find {index_file} or {safe_index_file} or {safe_peft_file}")

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))
    loader = safe_load_file if load_safe else partial(paddlenlp_load, map_location="np" if return_numpy else "cpu")

    ret = {}
    for shard_file in tqdm(shard_files):
        state_dict = loader(os.path.join(folder, shard_file))
        ret.update(state_dict)

    if not return_numpy:
        for key in list(ret.keys()):
            if isinstance(ret[key], np.ndarray):
                ret[key] = paddle.Tensor(ret.pop(key), zero_copy=True)

    return ret


def load_tp_checkpoint(folder, cls, config, return_numpy=False):
    """

    This load is performed efficiently: Load tp checkpoint only from cpu, no need to init the model.

    Args:
        folder (`str` or `os.PathLike`): A path to a folder containing the model checkpoint.
        cls (`str`): The model class.
        config (`AutoConfig`): The model config.
        return_numpy (bool): Whether load the tp checkpoint as numpy.
    """

    config = AutoConfig.from_pretrained(folder)
    if config.tensor_parallel_degree == 1:
        return load_sharded_checkpoint(folder, return_numpy=return_numpy)
    else:
        rank_model_path = os.path.join(folder, f"model_state.tp0{config.tensor_parallel_rank}.pdparams")
        model_path = os.path.join(folder, "model_state.pdparams")
        safe_model_path = os.path.join(folder, "model.safetensors")
        if os.path.exists(rank_model_path):
            return paddle.load(rank_model_path, return_numpy=return_numpy)
        elif os.path.exists(model_path):
            state_dict = cls.convert_tensor_parallel(model_path, config)
        elif os.path.exists(safe_model_path):
            with safe_open(safe_model_path, framework="np", device="cpu") as f:
                loaded_keys = f.keys()
            tp_actions = cls.get_tensor_parallel_convert_actions(config, loaded_keys)
            state_dict = load_state_dict(safe_model_path, tp_actions)
        else:  # shard files safetensors
            resolved_archive_file, resolved_sharded_files, sharded_metadata, is_sharded = cls._resolve_model_file_path(
                pretrained_model_name_or_path=folder,
                use_safetensors=True,
            )
            if len(resolved_sharded_files) > 1:
                resolved_sharded_files = tqdm(resolved_sharded_files, desc="Loading checkpoint shards")
            loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
            tp_actions = cls.get_tensor_parallel_convert_actions(config, loaded_state_dict_keys, ignore_error=True)
            state_dict = {}
            for shard_file in resolved_sharded_files:
                shard_state_dict = load_state_dict(
                    shard_file,
                    tp_actions,
                    loaded_state_dict_keys,
                )
                state_dict.update(shard_state_dict)
        if return_numpy:
            for k in list(state_dict.keys()):
                if not isinstance(state_dict[k], np.ndarray):
                    state_dict[k] = state_dict.pop(k).cpu().numpy()
    return state_dict
