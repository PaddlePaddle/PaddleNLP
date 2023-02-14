# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import os
from collections import defaultdict
from typing import Callable, Dict, Union

import paddle
import paddle.nn as nn

from .modeling_utils import _get_model_file, load_dict
from .models.cross_attention import LoRACrossAttnProcessor
from .utils import HF_CACHE, PPDIFFUSERS_CACHE, logging

logger = logging.get_logger(__name__)


LORA_WEIGHT_NAME = "paddle_lora_weights.pdparams"


class AttnProcsLayers(nn.Layer):
    def __init__(self, state_dict: Dict[str, paddle.Tensor]):
        super().__init__()
        self.layers = nn.LayerList(state_dict.values())
        self.mapping = {k: v for k, v in enumerate(state_dict.keys())}
        self.rev_mapping = {v: k for k, v in enumerate(state_dict.keys())}

        # we add a hook to state_dict() and load_state_dict() so that the
        # naming fits with `unet.attn_processors`
        def map_to(state_dict, *args, **kwargs):
            new_state_dict = {}
            for key, value in state_dict.items():
                num = int(key.split(".")[1])  # 0 is always "layers"
                new_key = key.replace(f"layers.{num}", self.mapping[num])
                new_state_dict[new_key] = value

            return new_state_dict

        def map_from(module, state_dict, *args, **kwargs):
            all_keys = list(state_dict.keys())
            for key in all_keys:
                replace_key = key.split(".processor")[0] + ".processor"
                new_key = key.replace(replace_key, f"layers.{module.rev_mapping[replace_key]}")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        self.register_state_dict_hook(map_to)
        self.register_load_state_dict_pre_hook(map_from, with_module=True)


class UNet2DConditionLoadersMixin:
    def load_attn_procs(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, paddle.Tensor]], **kwargs):
        r"""
        Load pretrained attention processor layers into `UNet2DConditionModel`. Attention processor layers have to be
        defined in
        [cross_attention.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py)
        and be a `paddle.nn.Layer` class.
        <Tip warning={true}>
            This function is experimental and might change in the future
        </Tip>
        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:
                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids should have an organization name, like `google/ddpm-celebahq-256`.
                    - A path to a *directory* containing model weights saved using [`~ModelMixin.save_config`], e.g.,
                      `./my_model_directory/`.
                    - A [paddle state
                      dict].
            from_hf_hub (bool, optional): whether to load from Huggingface Hub.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            subfolder (`str`, *optional*, defaults to `None`):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.
        """

        from_hf_hub = kwargs.pop("from_hf_hub", False)
        if from_hf_hub:
            cache_dir = kwargs.pop("cache_dir", HF_CACHE)
        else:
            cache_dir = kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", LORA_WEIGHT_NAME)

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                subfolder=subfolder,
                from_hf_hub=from_hf_hub,
            )
            state_dict = load_dict(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        # fill attn processors
        attn_processors = {}

        is_lora = all("lora" in k for k in state_dict.keys())

        if is_lora:
            lora_grouped_dict = defaultdict(dict)
            for key, value in state_dict.items():
                attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                lora_grouped_dict[attn_processor_key][sub_key] = value

            for key, value_dict in lora_grouped_dict.items():
                rank = value_dict["to_k_lora.down.weight"].shape[1]  # 0 -> 1, torch vs paddle nn.Linear
                cross_attention_dim = value_dict["to_k_lora.down.weight"].shape[0]  # 1 -> 0, torch vs paddle nn.Linear
                hidden_size = value_dict["to_k_lora.up.weight"].shape[1]  # 0 -> 1, torch vs paddle nn.Linear

                attn_processors[key] = LoRACrossAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank
                )
                attn_processors[key].load_dict(value_dict)

        else:
            raise ValueError(f"{model_file} does not seem to be in the correct format expected by LoRA training.")

        # set correct dtype & device
        attn_processors = {k: v.to(dtype=self.dtype) for k, v in attn_processors.items()}

        # set layers
        self.set_attn_processor(attn_processors)

    def save_attn_procs(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        weights_name: str = LORA_WEIGHT_NAME,
        save_function: Callable = None,
    ):
        r"""
        Save an attention procesor to a directory, so that it can be re-loaded using the
        `[`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`]` method.
        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            weights_name (`str`, *optional*, defaults to `LORA_WEIGHT_NAME`):
                The name of weights.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if save_function is None:
            save_function = paddle.save

        os.makedirs(save_directory, exist_ok=True)

        model_to_save = AttnProcsLayers(self.attn_processors)

        # Save the model
        state_dict = model_to_save.state_dict()

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".pdparams", "")
            if filename.startswith(weights_no_suffix) and os.path.isfile(full_filename) and is_main_process:
                os.remove(full_filename)

        # Save the model
        save_function(state_dict, os.path.join(save_directory, weights_name))

        logger.info(f"Model weights saved in {os.path.join(save_directory, weights_name)}")
