# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import Callable, Dict, Optional, Union

import paddle
import paddle.nn as nn

from .models.cross_attention import LoRACrossAttnProcessor
from .models.modeling_utils import convert_state_dict
from .utils import (
    DIFFUSERS_CACHE,
    FROM_DIFFUSERS,
    FROM_HF_HUB,
    HF_HUB_OFFLINE,
    PPDIFFUSERS_CACHE,
    TO_DIFFUSERS,
    _add_variant,
    _get_model_file,
    is_safetensors_available,
    is_torch_available,
    logging,
    smart_load,
)

logger = logging.get_logger(__name__)

if is_torch_available():
    import torch
if is_safetensors_available():
    import safetensors

TORCH_LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
TORCH_SAFETENSORS_LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"
PADDLE_LORA_WEIGHT_NAME = "paddle_lora_weights.pdparams"


def transpose_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if v.ndim == 2:
            new_state_dict[k] = v.T.contiguous() if hasattr(v, "contiguous") else v.T
        else:
            new_state_dict[k] = v.contiguous() if hasattr(v, "contiguous") else v
    return new_state_dict


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
        and be a `nn.Layer` class.

        <Tip warning={true}>

            This function is experimental and might change in the future.

        </Tip>

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids should have an organization name, like `google/ddpm-celebahq-256`.
                    - A path to a *directory* containing model weights saved using [`~ModelMixin.save_config`], e.g.,
                      `./my_model_directory/`.
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `diffusers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.
            from_hf_hub (bool, optional): whether to load from Huggingface Hub.
            from_diffusers (`bool`, *optional*, defaults to `False`):
                Load the model weights from a torch checkpoint save file.
        <Tip>

         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models).

        </Tip>

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use
        this method in a firewalled environment.

        </Tip>
        """
        from_hf_hub = kwargs.pop("from_hf_hub", FROM_HF_HUB)
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        from_diffusers = kwargs.pop("from_diffusers", FROM_DIFFUSERS)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        variant = kwargs.pop("variant", None)
        weights_name = kwargs.pop("weights_name", None)

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "paddle",
        }
        model_file = None
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            if from_diffusers:
                if is_safetensors_available():
                    try:
                        model_file = _get_model_file(
                            pretrained_model_name_or_path_or_dict,
                            weights_name=weights_name or _add_variant(TORCH_SAFETENSORS_LORA_WEIGHT_NAME, variant),
                            cache_dir=cache_dir,
                            force_download=force_download,
                            resume_download=resume_download,
                            proxies=proxies,
                            local_files_only=local_files_only,
                            use_auth_token=use_auth_token,
                            revision=revision,
                            subfolder=subfolder,
                            user_agent=user_agent,
                            from_hf_hub=from_hf_hub,
                        )
                    except Exception:  # noqa: E722
                        pass
                if model_file is None:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=weights_name or _add_variant(TORCH_LORA_WEIGHT_NAME, variant),
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        from_hf_hub=from_hf_hub,
                    )
            else:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weights_name or _add_variant(PADDLE_LORA_WEIGHT_NAME, variant),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    from_hf_hub=from_hf_hub,
                )
            assert model_file is not None
            state_dict = smart_load(model_file)

        else:
            state_dict = pretrained_model_name_or_path_or_dict

        # fill attn processors
        attn_processors = {}

        is_lora = all("lora" in k for k in state_dict.keys())

        if from_diffusers:
            state_dict = transpose_state_dict(state_dict)

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
        weights_name: str = PADDLE_LORA_WEIGHT_NAME,
        save_function: Callable = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        to_diffusers: Optional[bool] = None,
    ):
        r"""
        Save an attention processor to a directory, so that it can be re-loaded using the
        `[`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`]` method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `paddle.save` by another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
            to_diffusers (`bool`, *optional*, defaults to `None`):
                If specified, weights are saved in the format of torch. eg. linear need transpose.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Only when `to_diffusers` is True, Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
        """
        if to_diffusers is None:
            to_diffusers = TO_DIFFUSERS
        if to_diffusers and safe_serialization and not is_safetensors_available():
            raise ImportError("`safe_serialization` requires the `safetensors library: `pip install safetensors`.")

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        model_to_save = AttnProcsLayers(self.attn_processors)

        # Save the model
        state_dict = model_to_save.state_dict()

        # choose save_function
        if save_function is None:
            if to_diffusers:
                if safe_serialization:
                    if is_torch_available():
                        save_function = safetensors.torch.save_file
                        state_dict = convert_state_dict(state_dict, framework="torch")
                    else:
                        save_function = safetensors.numpy.save_file
                        state_dict = convert_state_dict(state_dict, framework="numpy")
                    weights_name = _add_variant(TORCH_SAFETENSORS_LORA_WEIGHT_NAME, variant)
                else:
                    if not is_torch_available():
                        raise ImportError(
                            "`to_diffusers=True` with `safe_serialization=False` requires the `torch library: `pip install torch`."
                        )
                    save_function = torch.save
                    weights_name = _add_variant(TORCH_LORA_WEIGHT_NAME, variant)
                    state_dict = convert_state_dict(state_dict, framework="torch")
                state_dict = transpose_state_dict(state_dict)
            else:
                save_function = paddle.save
                weights_name = _add_variant(PADDLE_LORA_WEIGHT_NAME, variant)

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            if to_diffusers:
                weights_no_suffix = (
                    weights_name.replace(".safetensors", "")
                    if safe_serialization
                    else weights_name.replace(".bin", "")
                )
            else:
                weights_no_suffix = weights_name.replace(".pdparams", "")
            if filename.startswith(weights_no_suffix) and os.path.isfile(full_filename) and is_main_process:
                os.remove(full_filename)

        # Save the model
        save_function(state_dict, os.path.join(save_directory, weights_name))

        logger.info(f"Model weights saved in {os.path.join(save_directory, weights_name)}")
