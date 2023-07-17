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
import copy
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import paddle
import paddle.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.file_download import _request_wrapper, hf_raise_for_status

from .models.attention_processor import (
    CustomDiffusionAttnProcessor,
    CustomDiffusionXFormersAttnProcessor,
    LoRAAttnProcessor,
)
from .models.modeling_utils import convert_state_dict
from .utils import (
    DIFFUSERS_CACHE,
    FROM_DIFFUSERS,
    FROM_HF_HUB,
    HF_HUB_OFFLINE,
    PPDIFFUSERS_CACHE,
    TEXT_ENCODER_ATTN_MODULE,
    TO_DIFFUSERS,
    _get_model_file,
    is_paddlenlp_available,
    is_safetensors_available,
    is_torch_available,
    is_torch_file,
    logging,
    ppdiffusers_url_download,
    safetensors_load,
    smart_load,
    torch_load,
)

logger = logging.get_logger(__name__)

if is_torch_available():
    import torch
if is_safetensors_available():
    import safetensors

if is_paddlenlp_available():
    from paddlenlp.transformers import PretrainedModel, PretrainedTokenizer

TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"

TORCH_LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
TORCH_LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"
PADDLE_LORA_WEIGHT_NAME = "paddle_lora_weights.pdparams"

TORCH_TEXT_INVERSION_NAME = "learned_embeds.bin"
TORCH_TEXT_INVERSION_NAME_SAFE = "learned_embeds.safetensors"
PADDLE_TEXT_INVERSION_NAME = "learned_embeds.pdparams"

TORCH_CUSTOM_DIFFUSION_WEIGHT_NAME = "pytorch_custom_diffusion_weights.bin"
TORCH_CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE = "pytorch_custom_diffusion_weights.safetensors"
PADDLE_CUSTOM_DIFFUSION_WEIGHT_NAME = "paddle_custom_diffusion_weights.pdparams"


def transpose_state_dict(state_dict, name_mapping=None):
    new_state_dict = {}
    for k, v in state_dict.items():
        if name_mapping is not None:
            for old_name, new_name in name_mapping.items():
                k = k.replace(old_name, new_name)
        if v.ndim == 2:
            new_state_dict[k] = v.T.contiguous() if hasattr(v, "contiguous") else v.T
        else:
            new_state_dict[k] = v.contiguous() if hasattr(v, "contiguous") else v
    return new_state_dict


class AttnProcsLayers(nn.Layer):
    def __init__(self, state_dict: Dict[str, paddle.Tensor]):
        super().__init__()
        self.layers = nn.LayerList(state_dict.values())
        self.mapping = dict(enumerate(state_dict.keys()))
        self.rev_mapping = {v: k for k, v in enumerate(state_dict.keys())}

        # .processor for unet, .self_attn for text encoder
        self.split_keys = [".processor", ".self_attn"]

        # we add a hook to state_dict() and load_state_dict() so that the
        # naming fits with `unet.attn_processors`
        def map_to(state_dict, *args, **kwargs):
            new_state_dict = {}
            for key, value in state_dict.items():
                num = int(key.split(".")[1])  # 0 is always "layers"
                new_key = key.replace(f"layers.{num}", self.mapping[num])
                new_state_dict[new_key] = value

            return new_state_dict

        def remap_key(key, state_dict):
            for k in self.split_keys:
                if k in key:
                    return key.split(k)[0] + k

            raise ValueError(
                f"There seems to be a problem with the state_dict: {set(state_dict.keys())}. {key} has to have one of {self.split_keys}."
            )

        def map_from(module, state_dict, *args, **kwargs):
            all_keys = list(state_dict.keys())
            for key in all_keys:
                replace_key = remap_key(key, state_dict)
                new_key = key.replace(replace_key, f"layers.{module.rev_mapping[replace_key]}")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        self.register_state_dict_hook(map_to)
        self.register_load_state_dict_pre_hook(map_from, with_module=True)


class UNet2DConditionLoadersMixin:
    text_encoder_name = TEXT_ENCODER_NAME
    unet_name = UNET_NAME

    def load_attn_procs(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, paddle.Tensor]], **kwargs):
        r"""
        Load pretrained attention processor layers into `UNet2DConditionModel`. Attention processor layers have to be
        defined in
        [`cross_attention.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py)
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
            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
            from_hf_hub (bool, optional): whether to load from Huggingface Hub.
            from_diffusers (`bool`, *optional*, defaults to `False`):
                Load the model weights from a torch checkpoint save file.
        <Tip>
         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models).
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
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        network_alpha = kwargs.pop("network_alpha", None)

        if from_diffusers and use_safetensors and not is_safetensors_available():
            raise ValueError(
                "`use_safetensors`=True but safetensors is not installed. Please install safetensors with `pip install safetenstors"
            )
        if use_safetensors is None:
            use_safetensors = is_safetensors_available()

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch" if from_diffusers else "paddle",
        }

        model_file = None

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            if from_diffusers:
                # Let's first try to load .safetensors weights
                if (use_safetensors and weight_name is None) or (
                    weight_name is not None and weight_name.endswith(".safetensors")
                ):
                    try:
                        model_file = _get_model_file(
                            pretrained_model_name_or_path_or_dict,
                            weights_name=weight_name or TORCH_LORA_WEIGHT_NAME_SAFE,
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
                        state_dict = smart_load(model_file)
                    except Exception:
                        model_file = None
                        pass
                if model_file is None:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=weight_name or TORCH_LORA_WEIGHT_NAME,
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
                    state_dict = smart_load(model_file)
            else:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or PADDLE_LORA_WEIGHT_NAME,
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
                state_dict = smart_load(model_file)
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        # fill attn processors
        attn_processors = {}

        is_lora = all("lora" in k for k in state_dict.keys())
        is_custom_diffusion = any("custom_diffusion" in k for k in state_dict.keys())

        if from_diffusers or is_torch_file(model_file):
            state_dict = transpose_state_dict(state_dict)

        if is_lora:
            is_new_lora_format = all(
                key.startswith(self.unet_name) or key.startswith(self.text_encoder_name) for key in state_dict.keys()
            )
            if is_new_lora_format:
                # Strip the `"unet"` prefix.
                is_text_encoder_present = any(key.startswith(self.text_encoder_name) for key in state_dict.keys())
                if is_text_encoder_present:
                    warn_message = "The state_dict contains LoRA params corresponding to the text encoder which are not being used here. To use both UNet and text encoder related LoRA params, use [`pipe.load_lora_weights()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraLoaderMixin.load_lora_weights)."
                    warnings.warn(warn_message)
                unet_keys = [k for k in state_dict.keys() if k.startswith(self.unet_name)]
                state_dict = {k.replace(f"{self.unet_name}.", ""): v for k, v in state_dict.items() if k in unet_keys}

            lora_grouped_dict = defaultdict(dict)
            for key, value in state_dict.items():
                attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                lora_grouped_dict[attn_processor_key][sub_key] = value.cast(
                    dtype="float32"
                )  # we must cast this to float32

            for key, value_dict in lora_grouped_dict.items():
                rank = value_dict["to_k_lora.down.weight"].shape[1]  # 0 -> 1, torch vs paddle nn.Linear
                cross_attention_dim = value_dict["to_k_lora.down.weight"].shape[0]  # 1 -> 0, torch vs paddle nn.Linear
                hidden_size = value_dict["to_k_lora.up.weight"].shape[1]  # 0 -> 1, torch vs paddle nn.Linear

                attn_processors[key] = LoRAAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=rank,
                    network_alpha=network_alpha,
                )
                attn_processors[key].load_dict(value_dict)
        elif is_custom_diffusion:
            custom_diffusion_grouped_dict = defaultdict(dict)
            for key, value in state_dict.items():
                if len(value) == 0:
                    custom_diffusion_grouped_dict[key] = {}
                else:
                    if "to_out" in key:
                        attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                    else:
                        attn_processor_key, sub_key = ".".join(key.split(".")[:-2]), ".".join(key.split(".")[-2:])
                    custom_diffusion_grouped_dict[attn_processor_key][sub_key] = value.cast(
                        dtype="float32"
                    )  # we must cast this to float32

            for key, value_dict in custom_diffusion_grouped_dict.items():
                if len(value_dict) == 0:
                    attn_processors[key] = CustomDiffusionAttnProcessor(
                        train_kv=False, train_q_out=False, hidden_size=None, cross_attention_dim=None
                    )
                else:
                    cross_attention_dim = value_dict["to_k_custom_diffusion.weight"].shape[
                        0
                    ]  # 1 -> 0, torch vs paddle nn.Linear
                    hidden_size = value_dict["to_k_custom_diffusion.weight"].shape[
                        1
                    ]  # 0 -> 1, torch vs paddle nn.Linear
                    train_q_out = True if "to_q_custom_diffusion.weight" in value_dict else False
                    attn_processors[key] = CustomDiffusionAttnProcessor(
                        train_kv=True,
                        train_q_out=train_q_out,
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                    )
                    attn_processors[key].load_dict(value_dict)
        else:
            raise ValueError(
                f"{model_file} does not seem to be in the correct format expected by LoRA or Custom Diffusion training."
            )
        # set correct dtype & device
        attn_processors = {k: v.to(dtype=self.dtype) for k, v in attn_processors.items()}

        # set layers
        self.set_attn_processor(attn_processors)

    def save_attn_procs(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = False,
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
            safe_serialization (`bool`, *optional*, defaults to `False`):
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

        is_custom_diffusion = any(
            isinstance(x, (CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor))
            for (_, x) in self.attn_processors.items()
        )
        if is_custom_diffusion:
            model_to_save = AttnProcsLayers(
                {
                    y: x
                    for (y, x) in self.attn_processors.items()
                    if isinstance(x, (CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor))
                }
            )
            state_dict = model_to_save.state_dict()
            for name, attn in self.attn_processors.items():
                if len(attn.state_dict()) == 0:
                    state_dict[name] = {}
        else:
            model_to_save = AttnProcsLayers(self.attn_processors)
            state_dict = model_to_save.state_dict()

        if weight_name is None:
            if to_diffusers:
                if safe_serialization:
                    weight_name = (
                        TORCH_CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE if is_custom_diffusion else TORCH_LORA_WEIGHT_NAME_SAFE
                    )
                else:
                    weight_name = TORCH_CUSTOM_DIFFUSION_WEIGHT_NAME if is_custom_diffusion else TORCH_LORA_WEIGHT_NAME
            else:
                weight_name = PADDLE_CUSTOM_DIFFUSION_WEIGHT_NAME if is_custom_diffusion else PADDLE_LORA_WEIGHT_NAME

        # choose save_function
        if save_function is None:
            if to_diffusers:
                if safe_serialization:
                    if is_torch_available():
                        _save_function = safetensors.torch.save_file
                        state_dict = convert_state_dict(state_dict, framework="torch")
                    else:
                        _save_function = safetensors.numpy.save_file
                        state_dict = convert_state_dict(state_dict, framework="numpy")

                    def save_function(weights, filename):
                        return _save_function(weights, filename, metadata={"format": "pt"})

                else:
                    if not is_torch_available():
                        raise ImportError(
                            "`to_diffusers=True` with `safe_serialization=False` requires the `torch library: `pip install torch`."
                        )
                    save_function = torch.save
                    state_dict = convert_state_dict(state_dict, framework="torch")
                state_dict = transpose_state_dict(state_dict)
            else:
                save_function = paddle.save

        # Save the model
        save_function(state_dict, os.path.join(save_directory, weight_name))

        logger.info(f"Model weights saved in {os.path.join(save_directory, weight_name)}")


class TextualInversionLoaderMixin:
    r"""
    Mixin class for loading textual inversion tokens and embeddings to the tokenizer and text encoder.
    """

    def maybe_convert_prompt(self, prompt: Union[str, List[str]], tokenizer: "PretrainedTokenizer"):
        r"""
        Maybe convert a prompt into a "multi vector"-compatible prompt. If the prompt includes a token that corresponds
        to a multi-vector textual inversion embedding, this function will process the prompt so that the special token
        is replaced with multiple special tokens each corresponding to one of the vectors. If the prompt has no textual
        inversion token or a textual inversion token that is a single vector, the input prompt is simply returned.
        Parameters:
            prompt (`str` or list of `str`):
                The prompt or prompts to guide the image generation.
            tokenizer (`PretrainedTokenizer`):
                The tokenizer responsible for encoding the prompt into input tokens.
        Returns:
            `str` or list of `str`: The converted prompt
        """
        if not isinstance(prompt, List):
            prompts = [prompt]
        else:
            prompts = prompt

        prompts = [self._maybe_convert_prompt(p, tokenizer) for p in prompts]

        if not isinstance(prompt, List):
            return prompts[0]

        return prompts

    def _maybe_convert_prompt(self, prompt: str, tokenizer: "PretrainedTokenizer"):
        r"""
        Maybe convert a prompt into a "multi vector"-compatible prompt. If the prompt includes a token that corresponds
        to a multi-vector textual inversion embedding, this function will process the prompt so that the special token
        is replaced with multiple special tokens each corresponding to one of the vectors. If the prompt has no textual
        inversion token or a textual inversion token that is a single vector, the input prompt is simply returned.
        Parameters:
            prompt (`str`):
                The prompt to guide the image generation.
            tokenizer (`PretrainedTokenizer`):
                The tokenizer responsible for encoding the prompt into input tokens.
        Returns:
            `str`: The converted prompt
        """
        tokens = tokenizer.tokenize(prompt)
        for token in tokens:
            if token in tokenizer.added_tokens_encoder:
                replacement = token
                i = 1
                while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                    replacement += f" {token}_{i}"
                    i += 1

                prompt = prompt.replace(token, replacement)

        return prompt

    def load_textual_inversion(
        self,
        pretrained_model_name_or_path: Union[str, Dict[str, paddle.Tensor]],
        token: Optional[str] = None,
        **kwargs
    ):
        r"""
        Load textual inversion embeddings into the text encoder of stable diffusion pipelines. Both `diffusers` and
        `Automatic1111` formats are supported (see example below).
        <Tip warning={true}>
        This function is experimental and might change in the future.
        </Tip>
        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:
                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids should have an organization name, like
                      `"sd-concepts-library/low-poly-hd-logos-icons"`.
                    - A path to a *directory* containing textual inversion weights, e.g.
                      `./my_text_inversion_directory/`.
            weight_name (`str`, *optional*):
                Name of a custom weight file. This should be used in two cases:
                    - The saved textual inversion file is in `diffusers` format, but was saved under a specific weight
                      name, such as `text_inv.bin`.
                    - The saved textual inversion file is in the "Automatic1111" form.
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
            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
        <Tip>
         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models).
        </Tip>
        Example:
        To load a textual inversion embedding vector in `ppdiffusers` format:
        ```py
        from ppdiffusers import StableDiffusionPipeline
        import paddle
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, paddle_dtype=paddle.float16)
        pipe.load_textual_inversion("sd-concepts-library/cat-toy")
        prompt = "A <cat-toy> backpack"
        image = pipe(prompt, num_inference_steps=50).images[0]
        image.save("cat-backpack.png")
        ```
        To load a textual inversion embedding vector in Automatic1111 format, make sure to first download the vector,
        e.g. from [civitAI](https://civitai.com/models/3036?modelVersionId=9857) and then load the vector locally:
        ```py
        from diffusers import StableDiffusionPipeline
        import paddle
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, paddle_dtype=paddle.float16)
        pipe.load_textual_inversion("./charturnerv2.pt", token="charturnerv2")
        prompt = "charturnerv2, multiple views of the same character in the same outfit, a character turnaround of a woman wearing a black jacket and red shirt, best quality, intricate details."
        image = pipe(prompt, num_inference_steps=50).images[0]
        image.save("character.png")
        ```
        """
        if not hasattr(self, "tokenizer") or not isinstance(self.tokenizer, PretrainedTokenizer):
            raise ValueError(
                f"{self.__class__.__name__} requires `self.tokenizer` of type `PretrainedTokenizer` for calling"
                f" `{self.load_textual_inversion.__name__}`"
            )

        if not hasattr(self, "text_encoder") or not isinstance(self.text_encoder, PretrainedModel):
            raise ValueError(
                f"{self.__class__.__name__} requires `self.text_encoder` of type `PretrainedModel` for calling"
                f" `{self.load_textual_inversion.__name__}`"
            )
        from_hf_hub = kwargs.pop("from_hf_hub", FROM_HF_HUB)
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )
        from_diffusers = kwargs.pop("from_diffusers", FROM_DIFFUSERS)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        if from_diffusers and use_safetensors and not is_safetensors_available():
            raise ValueError(
                "`use_safetensors`=True but safetensors is not installed. Please install safetensors with `pip install safetenstors"
            )
        if use_safetensors is None:
            use_safetensors = is_safetensors_available()
        user_agent = {
            "file_type": "text_inversion",
            "framework": "pytorch" if from_diffusers else "paddle",
        }

        # 1. Load textual inversion file
        model_file = None
        # Let's first try to load .safetensors weights
        if from_diffusers:
            if (use_safetensors and weight_name is None) or (
                weight_name is not None and weight_name.endswith(".safetensors")
            ):
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=weight_name or TORCH_TEXT_INVERSION_NAME_SAFE,
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
                    state_dict = safetensors_load(model_file)
                except Exception:
                    model_file = None
                    pass
            if model_file is None:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=weight_name or TORCH_TEXT_INVERSION_NAME,
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
                state_dict = torch_load(model_file)
        else:
            model_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=weight_name or PADDLE_TEXT_INVERSION_NAME,
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
            if is_torch_file(model_file):
                try:
                    state_dict = safetensors_load(model_file)
                except:
                    state_dict = torch_load(model_file)
            else:
                state_dict = paddle.load(model_file)

        # 2. Load token and embedding correcly from file
        if isinstance(state_dict, paddle.Tensor):
            if token is None:
                raise ValueError(
                    "You are trying to load a textual inversion embedding that has been saved as a Paddle tensor. Make sure to pass the name of the corresponding token in this case: `token=...`."
                )
            embedding = state_dict
        elif len(state_dict) == 1:
            # diffusers
            loaded_token, embedding = next(iter(state_dict.items()))
        elif "string_to_param" in state_dict:
            # A1111
            loaded_token = state_dict["name"]
            embedding = state_dict["string_to_param"]["*"]

        if token is not None and loaded_token != token:
            logger.warn(f"The loaded token: {loaded_token} is overwritten by the passed token {token}.")
        else:
            token = loaded_token

        if not isinstance(state_dict, paddle.Tensor):
            if hasattr(embedding, "detach"):
                embedding = embedding.detach()
            if hasattr(embedding, "cpu"):
                embedding = embedding.cpu()
            if hasattr(embedding, "numpy"):
                embedding = embedding.numpy()
            embedding = paddle.to_tensor(embedding)
        embedding = embedding.cast(dtype=self.text_encoder.dtype)

        # 3. Make sure we don't mess up the tokenizer or text encoder
        vocab = self.tokenizer.get_vocab()
        if token in vocab:
            raise ValueError(
                f"Token {token} already in tokenizer vocabulary. Please choose a different token name or remove {token} and embedding from the tokenizer and text encoder."
            )
        elif f"{token}_1" in vocab:
            multi_vector_tokens = [token]
            i = 1
            while f"{token}_{i}" in self.tokenizer.added_tokens_encoder:
                multi_vector_tokens.append(f"{token}_{i}")
                i += 1

            raise ValueError(
                f"Multi-vector Token {multi_vector_tokens} already in tokenizer vocabulary. Please choose a different token name or remove the {multi_vector_tokens} and embedding from the tokenizer and text encoder."
            )

        is_multi_vector = len(embedding.shape) > 1 and embedding.shape[0] > 1

        if is_multi_vector:
            tokens = [token] + [f"{token}_{i}" for i in range(1, embedding.shape[0])]
            embeddings = [e for e in embedding]  # noqa: C416
        else:
            tokens = [token]
            embeddings = [embedding[0]] if len(embedding.shape) > 1 else [embedding]

        # add tokens and get ids
        self.tokenizer.add_tokens(tokens)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # resize token embeddings and set new embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        with paddle.no_grad():
            for token_id, embedding in zip(token_ids, embeddings):
                self.text_encoder.get_input_embeddings().weight[token_id] = embedding

        logger.info(f"Loaded textual inversion embedding for {token}.")


class LoraLoaderMixin:
    r"""
    Utility class for handling the loading LoRA layers into UNet (of class [`UNet2DConditionModel`]) and Text Encoder
    (of class [`CLIPTextModel`](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel)).
    <Tip warning={true}>
    This function is experimental and might change in the future.
    </Tip>
    """
    text_encoder_name = TEXT_ENCODER_NAME
    unet_name = UNET_NAME

    def load_lora_weights(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, paddle.Tensor]], **kwargs):
        r"""
        Load pretrained attention processor layers (such as LoRA) into [`UNet2DConditionModel`] and
        [`CLIPTextModel`](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel)).
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
            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
        <Tip>
        It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
        models](https://huggingface.co/docs/hub/models-gated#gated-models).
        </Tip>
        """
        # Load the main state dict first which has the LoRA layers for either of
        # UNet and text encoder or both.
        from_hf_hub = kwargs.pop("from_hf_hub", FROM_HF_HUB)
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )
        from_diffusers = kwargs.pop("from_diffusers", FROM_DIFFUSERS)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        # set lora scale to a reasonable default
        self._lora_scale = 1.0

        if from_diffusers and use_safetensors and not is_safetensors_available():
            raise ValueError(
                "`use_safetensors`=True but safetensors is not installed. Please install safetensors with `pip install safetenstors"
            )
        if use_safetensors is None:
            use_safetensors = is_safetensors_available()

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch" if from_diffusers else "paddle",
        }

        model_file = None

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            if from_diffusers:
                # Let's first try to load .safetensors weights
                if (use_safetensors and weight_name is None) or (
                    weight_name is not None and weight_name.endswith(".safetensors")
                ):
                    try:
                        model_file = _get_model_file(
                            pretrained_model_name_or_path_or_dict,
                            weights_name=weight_name or TORCH_LORA_WEIGHT_NAME_SAFE,
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
                        state_dict = smart_load(model_file)
                    except Exception:
                        model_file = None
                        pass
                if model_file is None:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=weight_name or TORCH_LORA_WEIGHT_NAME,
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
                    state_dict = smart_load(model_file)
            else:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or PADDLE_LORA_WEIGHT_NAME,
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
                state_dict = smart_load(model_file)
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        if not from_diffusers:
            from_diffusers = is_torch_file(model_file)

        # Convert kohya-ss Style LoRA attn procs to ppdiffusers attn procs
        network_alpha = None
        if all((k.startswith("lora_te_") or k.startswith("lora_unet_")) for k in state_dict.keys()):
            state_dict, network_alpha = self._convert_kohya_lora_to_diffusers(state_dict)
            from_diffusers = True

        # If the serialization format is new (introduced in https://github.com/huggingface/diffusers/pull/2918),
        # then the `state_dict` keys should have `self.unet_name` and/or `self.text_encoder_name` as
        # their prefixes.
        keys = list(state_dict.keys())
        if all(key.startswith(self.unet_name) or key.startswith(self.text_encoder_name) for key in keys):
            # Load the layers corresponding to UNet.
            unet_keys = [k for k in keys if k.startswith(self.unet_name)]
            logger.info(f"Loading {self.unet_name}.")
            unet_lora_state_dict = {
                k.replace(f"{self.unet_name}.", ""): v for k, v in state_dict.items() if k in unet_keys
            }
            self.unet.load_attn_procs(unet_lora_state_dict, network_alpha=network_alpha, from_diffusers=from_diffusers)

            # Load the layers corresponding to text encoder and make necessary adjustments.
            text_encoder_keys = [k for k in keys if k.startswith(self.text_encoder_name)]
            text_encoder_lora_state_dict = {
                k.replace(f"{self.text_encoder_name}.", ""): v for k, v in state_dict.items() if k in text_encoder_keys
            }
            if len(text_encoder_lora_state_dict) > 0:
                logger.info(f"Loading {self.text_encoder_name}.")
                attn_procs_text_encoder = self._load_text_encoder_attn_procs(
                    text_encoder_lora_state_dict,
                    network_alpha=network_alpha,
                    from_diffusers=from_diffusers,
                )
                self._modify_text_encoder(attn_procs_text_encoder)

                # save lora attn procs of text encoder so that it can be easily retrieved
                self._text_encoder_lora_attn_procs = attn_procs_text_encoder

        # Otherwise, we're dealing with the old format. This means the `state_dict` should only
        # contain the module names of the `unet` as its keys WITHOUT any prefix.
        elif not all(
            key.startswith(self.unet_name) or key.startswith(self.text_encoder_name) for key in state_dict.keys()
        ):
            self.unet.load_attn_procs(state_dict, network_alpha=network_alpha, from_diffusers=from_diffusers)
            warn_message = "You have saved the LoRA weights using the old format. To convert the old LoRA weights to the new format, you can first load them in a dictionary and then create a new dictionary like the following: `new_state_dict = {f'unet'.{module_name}: params for module_name, params in old_state_dict.items()}`."
            warnings.warn(warn_message)

    @property
    def lora_scale(self) -> float:
        # property function that returns the lora scale which can be set at run time by the pipeline.
        # if _lora_scale has not been set, return 1
        return self._lora_scale if hasattr(self, "_lora_scale") else 1.0

    @property
    def text_encoder_lora_attn_procs(self):
        if hasattr(self, "_text_encoder_lora_attn_procs"):
            return self._text_encoder_lora_attn_procs
        return

    def _remove_text_encoder_monkey_patch(self):
        # Loop over the nn.MultiHeadAttention module of text_encoder
        for name, attn_module in self.text_encoder.named_sublayers(include_self=True):
            if name.endswith(TEXT_ENCODER_ATTN_MODULE):
                # Loop over the LoRA layers
                for _, text_encoder_attr in self._lora_attn_processor_attr_to_text_encoder_attr.items():
                    # Retrieve the q/k/v/out projection of nn.MultiHeadAttention
                    module = attn_module.get_sublayer(text_encoder_attr)
                    if hasattr(module, "old_forward"):
                        # restore original `forward` to remove monkey-patch
                        module.forward = module.old_forward
                        delattr(module, "old_forward")

                # new added by Junnyu, no exists in diffusers
                if hasattr(attn_module, "processor"):
                    # del processor
                    delattr(attn_module, "processor")

    def _modify_text_encoder(self, attn_processors: Dict[str, LoRAAttnProcessor]):
        r"""
        Monkey-patches the forward passes of attention modules of the text encoder.

        Parameters:
            attn_processors: Dict[str, `LoRAAttnProcessor`]:
                A dictionary mapping the module names and their corresponding [`~LoRAAttnProcessor`].
        """

        # First, remove any monkey-patch that might have been applied before
        self._remove_text_encoder_monkey_patch()

        # Loop over the nn.MultiHeadAttention module of text_encoder
        for name, attn_module in self.text_encoder.named_sublayers(include_self=True):
            if name.endswith(TEXT_ENCODER_ATTN_MODULE):
                # Loop over the LoRA layers
                for attn_proc_attr, text_encoder_attr in self._lora_attn_processor_attr_to_text_encoder_attr.items():
                    # Retrieve the q/k/v/out projection of nn.MultiHeadAttention and its corresponding LoRA layer.
                    module = attn_module.get_sublayer(text_encoder_attr)
                    lora_layer = attn_processors[name].get_sublayer(attn_proc_attr)
                    # save old_forward to module that can be used to remove monkey-patch
                    old_forward = module.old_forward = module.forward

                    # create a new scope that locks in the old_forward, lora_layer value for each new_forward function
                    # for more detail, see https://github.com/huggingface/diffusers/pull/3490#issuecomment-1555059060
                    def make_new_forward(old_forward, lora_layer):
                        def new_forward(x):
                            result = old_forward(x) + self.lora_scale * lora_layer(x)
                            return result

                        return new_forward

                    # Monkey-patch.
                    module.forward = make_new_forward(old_forward, lora_layer)

                # new added by Junnyu, no exists in diffusers
                attn_module.processor = attn_processors[name]

    @property
    def _lora_attn_processor_attr_to_text_encoder_attr(self):
        return {
            "to_q_lora": "q_proj",
            "to_k_lora": "k_proj",
            "to_v_lora": "v_proj",
            "to_out_lora": "out_proj",
        }

    def _load_text_encoder_attn_procs(
        self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, paddle.Tensor]], **kwargs
    ):
        r"""
        Load pretrained attention processor layers for
        [`CLIPTextModel`](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel).
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
            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
        Returns:
            `Dict[name, LoRAAttnProcessor]`: Mapping between the module names and their corresponding
            [`LoRAAttnProcessor`].
        <Tip>
        It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
        models](https://huggingface.co/docs/hub/models-gated#gated-models).
        </Tip>
        """

        from_hf_hub = kwargs.pop("from_hf_hub", FROM_HF_HUB)
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )
        from_diffusers = kwargs.pop("from_diffusers", FROM_DIFFUSERS)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        network_alpha = kwargs.pop("network_alpha", None)

        if from_diffusers and use_safetensors and not is_safetensors_available():
            raise ValueError(
                "`use_safetensors`=True but safetensors is not installed. Please install safetensors with `pip install safetenstors"
            )
        if use_safetensors is None:
            use_safetensors = is_safetensors_available()
        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch" if from_diffusers else "paddle",
        }

        model_file = None
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            if from_diffusers:
                # Let's first try to load .safetensors weights
                if (use_safetensors and weight_name is None) or (
                    weight_name is not None and weight_name.endswith(".safetensors")
                ):
                    try:
                        model_file = _get_model_file(
                            pretrained_model_name_or_path_or_dict,
                            weights_name=weight_name or TORCH_LORA_WEIGHT_NAME_SAFE,
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
                        state_dict = smart_load(model_file)
                    except Exception:
                        model_file = None
                        pass
                if model_file is None:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=weight_name or TORCH_LORA_WEIGHT_NAME,
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
                    state_dict = smart_load(model_file)
            else:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or PADDLE_LORA_WEIGHT_NAME,
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
                state_dict = smart_load(model_file)
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        # fill attn processors
        attn_processors = {}

        is_lora = all("lora" in k for k in state_dict.keys())

        if from_diffusers or is_torch_file(model_file):
            state_dict = transpose_state_dict(state_dict, name_mapping={".encoder.": ".transformer."})

        if is_lora:
            lora_grouped_dict = defaultdict(dict)
            for key, value in state_dict.items():
                attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                lora_grouped_dict[attn_processor_key][sub_key] = value.cast(
                    dtype="float32"
                )  # we must cast this to float32

            for key, value_dict in lora_grouped_dict.items():
                rank = value_dict["to_k_lora.down.weight"].shape[1]  # 0 -> 1, torch vs paddle nn.Linear
                cross_attention_dim = value_dict["to_k_lora.down.weight"].shape[0]  # 1 -> 0, torch vs paddle nn.Linear
                hidden_size = value_dict["to_k_lora.up.weight"].shape[1]  # 0 -> 1, torch vs paddle nn.Linear

                attn_processors[key] = LoRAAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=rank,
                    network_alpha=network_alpha,
                )
                attn_processors[key].load_dict(value_dict)

        else:
            raise ValueError(f"{model_file} does not seem to be in the correct format expected by LoRA training.")

        # set correct dtype & device
        attn_processors = {k: v.to(dtype=self.text_encoder.dtype) for k, v in attn_processors.items()}
        return attn_processors

    @classmethod
    def save_lora_weights(
        self,
        save_directory: Union[str, os.PathLike],
        unet_lora_layers: Dict[str, nn.Layer] = None,
        text_encoder_lora_layers: Dict[str, nn.Layer] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = False,
        to_diffusers: Optional[bool] = None,
    ):
        r"""
        Save the LoRA parameters corresponding to the UNet and the text encoder.
        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            unet_lora_layers (`Dict[str, nn.Layer`]):
                State dict of the LoRA layers corresponding to the UNet. Specifying this helps to make the
                serialization process easier and cleaner.
            text_encoder_lora_layers (`Dict[str, nn.Layer`]):
                State dict of the LoRA layers corresponding to the `text_encoder`. Since the `text_encoder` comes from
                `paddlenlp`, we cannot rejig it. That is why we have to explicitly pass the text encoder LoRA state
                dict.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
        """
        if to_diffusers is None:
            to_diffusers = TO_DIFFUSERS
        if to_diffusers and safe_serialization and not is_safetensors_available():
            raise ImportError("`safe_serialization` requires the `safetensors library: `pip install safetensors`.")

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # Create a flat dictionary.
        state_dict = {}
        if unet_lora_layers is not None:
            unet_lora_state_dict = {
                f"{self.unet_name}.{module_name}": param
                for module_name, param in unet_lora_layers.state_dict().items()
            }
            state_dict.update(unet_lora_state_dict)
        if text_encoder_lora_layers is not None:
            text_encoder_lora_state_dict = {
                f"{self.text_encoder_name}.{module_name}": param
                for module_name, param in text_encoder_lora_layers.state_dict().items()
            }
            state_dict.update(text_encoder_lora_state_dict)
            # TODO junnyu, rename paramaters.

        # Save the model
        if weight_name is None:
            if to_diffusers:
                if safe_serialization:
                    weight_name = TORCH_LORA_WEIGHT_NAME_SAFE
                else:
                    weight_name = TORCH_LORA_WEIGHT_NAME
            else:
                weight_name = PADDLE_LORA_WEIGHT_NAME

        # choose save_function
        if save_function is None:
            if to_diffusers:
                if safe_serialization:
                    if is_torch_available():
                        _save_function = safetensors.torch.save_file
                        state_dict = convert_state_dict(state_dict, framework="torch")
                    else:
                        _save_function = safetensors.numpy.save_file
                        state_dict = convert_state_dict(state_dict, framework="numpy")

                    def save_function(weights, filename):
                        return _save_function(weights, filename, metadata={"format": "pt"})

                else:
                    if not is_torch_available():
                        raise ImportError(
                            "`to_diffusers=True` with `safe_serialization=False` requires the `torch library: `pip install torch`."
                        )
                    save_function = torch.save
                    state_dict = convert_state_dict(state_dict, framework="torch")
                state_dict = transpose_state_dict(state_dict, name_mapping={".transformer.": ".encoder."})
            else:
                save_function = paddle.save

        save_function(state_dict, os.path.join(save_directory, weight_name))
        logger.info(f"Model weights saved in {os.path.join(save_directory, weight_name)}")

    def _convert_kohya_lora_to_diffusers(self, state_dict):
        unet_state_dict = {}
        te_state_dict = {}
        network_alpha = None

        for key, value in state_dict.items():
            if "lora_down" in key:
                lora_name = key.split(".")[0]
                lora_name_up = lora_name + ".lora_up.weight"
                lora_name_alpha = lora_name + ".alpha"
                if lora_name_alpha in state_dict:
                    # we must cast this to float32, before get item
                    alpha = state_dict[lora_name_alpha].cast("float32").item()
                    if network_alpha is None:
                        network_alpha = alpha
                    elif network_alpha != alpha:
                        raise ValueError("Network alpha is not consistent")

                if lora_name.startswith("lora_unet_"):
                    diffusers_name = key.replace("lora_unet_", "").replace("_", ".")
                    diffusers_name = diffusers_name.replace("down.blocks", "down_blocks")
                    diffusers_name = diffusers_name.replace("mid.block", "mid_block")
                    diffusers_name = diffusers_name.replace("up.blocks", "up_blocks")
                    diffusers_name = diffusers_name.replace("transformer.blocks", "transformer_blocks")
                    diffusers_name = diffusers_name.replace("to.q.lora", "to_q_lora")
                    diffusers_name = diffusers_name.replace("to.k.lora", "to_k_lora")
                    diffusers_name = diffusers_name.replace("to.v.lora", "to_v_lora")
                    diffusers_name = diffusers_name.replace("to.out.0.lora", "to_out_lora")
                    if "transformer_blocks" in diffusers_name:
                        if "attn1" in diffusers_name or "attn2" in diffusers_name:
                            diffusers_name = diffusers_name.replace("attn1", "attn1.processor")
                            diffusers_name = diffusers_name.replace("attn2", "attn2.processor")
                            unet_state_dict[diffusers_name] = value
                            unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict[lora_name_up]
                elif lora_name.startswith("lora_te_"):
                    diffusers_name = key.replace("lora_te_", "").replace("_", ".")
                    diffusers_name = diffusers_name.replace("text.model", "text_model")
                    diffusers_name = diffusers_name.replace("self.attn", "self_attn")
                    diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
                    diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
                    diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
                    diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
                    if "self_attn" in diffusers_name:
                        te_state_dict[diffusers_name] = value
                        te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict[lora_name_up]

        unet_state_dict = {f"{UNET_NAME}.{module_name}": params for module_name, params in unet_state_dict.items()}
        te_state_dict = {f"{TEXT_ENCODER_NAME}.{module_name}": params for module_name, params in te_state_dict.items()}
        new_state_dict = {**unet_state_dict, **te_state_dict}
        return new_state_dict, network_alpha


class FromCkptMixin:
    """This helper class allows to directly load .ckpt or .safetensors stable diffusion file_extension
    into the respective classes."""

    @classmethod
    def from_ckpt(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Instantiate a Paddle diffusion pipeline from pre-trained pipeline weights saved in the original .ckpt format.
        The pipeline is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated).
        Parameters:
            pretrained_model_link_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A link to the .ckpt file on the Hub. Should be in the format
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>"`
                    - A link to the .safetensors file on the civitai.com. Should be in the format
                      `"https://civitai.com/api/download/models/<number_of_file>"`
                    - A path to a *file* containing all pipeline weights.
            paddle_dtype (`str` or `paddle.dtype`, *optional*):
                Override the default `paddle.dtype` and load the model under this dtype.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            extract_ema (`bool`, *optional*, defaults to `False`): Only relevant for
                checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights or not. Defaults
                to `False`. Pass `True` to extract the EMA weights. EMA weights usually yield higher quality images for
                inference. Non-EMA weights are usually better to continue fine-tuning.
            upcast_attention (`bool`, *optional*, defaults to `None`):
                Whether the attention computation should always be upcasted. This is necessary when running stable
            image_size (`int`, *optional*, defaults to 512):
                The image size that the model was trained on. Use 512 for Stable Diffusion v1.X and Stable Diffusion v2
                Base. Use 768 for Stable Diffusion v2.
            prediction_type (`str`, *optional*):
                The prediction type that the model was trained on. Use `'epsilon'` for Stable Diffusion v1.X and Stable
                Diffusion v2 Base. Use `'v_prediction'` for Stable Diffusion v2.
            num_in_channels (`int`, *optional*, defaults to None):
                The number of input channels. If `None`, it will be automatically inferred.
            scheduler_type (`str`, *optional*, defaults to 'pndm'):
                Type of scheduler to use. Should be one of `["pndm", "lms", "heun", "euler", "euler-ancestral", "dpm",
                "ddim"]`.
            load_safety_checker (`bool`, *optional*, defaults to `False`):
                Whether to load the safety checker or not. Defaults to `False`.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load - and saveable variables - *i.e.* the pipeline components - of the
                specific pipeline class. The overwritten components are then directly passed to the pipelines
                `__init__` method. See example below for more information.
        Examples:
        ```py
        >>> from ppdiffusers import StableDiffusionPipeline
        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = StableDiffusionPipeline.from_ckpt(
        ...     "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors"
        ... )
        >>> # Download pipeline from local file
        >>> # file is downloaded under ./v1-5-pruned-emaonly.ckpt
        >>> pipeline = StableDiffusionPipeline.from_ckpt("./v1-5-pruned-emaonly")
        >>> # Enable float16
        >>> pipeline = StableDiffusionPipeline.from_ckpt(
        ...     "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt",
        ...     paddle_dtype=paddle.float16,
        ... )
        ```
        """
        # import here to avoid circular dependency
        from .pipelines.stable_diffusion.convert_from_ckpt import (
            download_from_original_stable_diffusion_ckpt,
        )

        from_hf_hub = "huggingface.co" in pretrained_model_link_or_path or "hf.co"
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        extract_ema = kwargs.pop("extract_ema", False)
        image_size = kwargs.pop("image_size", 512)
        scheduler_type = kwargs.pop("scheduler_type", "pndm")
        num_in_channels = kwargs.pop("num_in_channels", None)
        upcast_attention = kwargs.pop("upcast_attention", None)
        load_safety_checker = kwargs.pop("load_safety_checker", False)
        prediction_type = kwargs.pop("prediction_type", None)

        paddle_dtype = kwargs.pop("paddle_dtype", None)

        pipeline_name = cls.__name__

        # TODO: For now we only support stable diffusion
        stable_unclip = None
        controlnet = False

        if pipeline_name == "StableDiffusionControlNetPipeline":
            model_type = "FrozenCLIPEmbedder"
            controlnet = True
        elif "StableDiffusion" in pipeline_name:
            model_type = "FrozenCLIPEmbedder"
        elif pipeline_name == "StableUnCLIPPipeline":
            model_type == "FrozenOpenCLIPEmbedder"
            stable_unclip = "txt2img"
        elif pipeline_name == "StableUnCLIPImg2ImgPipeline":
            model_type == "FrozenOpenCLIPEmbedder"
            stable_unclip = "img2img"
        elif pipeline_name == "PaintByExamplePipeline":
            model_type == "PaintByExample"
        elif pipeline_name == "LDMTextToImagePipeline":
            model_type == "LDMTextToImage"
        else:
            raise ValueError(f"Unhandled pipeline class: {pipeline_name}")

        pretrained_model_link_or_path = str(pretrained_model_link_or_path)
        if os.path.isfile(pretrained_model_link_or_path):
            checkpoint_path = pretrained_model_link_or_path
        elif pretrained_model_link_or_path.startswith("http://") or pretrained_model_link_or_path.startswith(
            "https://"
        ):
            # HF Hub models
            if any(p in pretrained_model_link_or_path for p in ["huggingface.co", "hf.co"]):
                # remove huggingface url
                for prefix in ["https://huggingface.co/", "huggingface.co/", "hf.co/", "https://hf.co/"]:
                    if pretrained_model_link_or_path.startswith(prefix):
                        pretrained_model_link_or_path = pretrained_model_link_or_path[len(prefix) :]

                # Code based on diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained
                ckpt_path = Path(pretrained_model_link_or_path)
                if not ckpt_path.is_file():
                    # get repo_id and (potentially nested) file path of ckpt in repo
                    repo_id = str(Path().joinpath(*ckpt_path.parts[:2]))
                    file_path = str(Path().joinpath(*ckpt_path.parts[2:]))

                    if file_path.startswith("blob/"):
                        file_path = file_path[len("blob/") :]

                    if file_path.startswith("main/"):
                        file_path = file_path[len("main/") :]

                    checkpoint_path = hf_hub_download(
                        repo_id,
                        filename=file_path,
                        cache_dir=cache_dir,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        force_download=force_download,
                    )
                else:
                    checkpoint_path = ckpt_path
            else:
                checkpoint_path = ppdiffusers_url_download(
                    pretrained_model_link_or_path,
                    cache_dir=cache_dir,
                    filename=http_file_name(pretrained_model_link_or_path).strip('"'),
                    force_download=force_download,
                    resume_download=resume_download,
                )
        else:
            checkpoint_path = pretrained_model_link_or_path

        pipe = download_from_original_stable_diffusion_ckpt(
            checkpoint_path,
            pipeline_class=cls,
            model_type=model_type,
            stable_unclip=stable_unclip,
            controlnet=controlnet,
            extract_ema=extract_ema,
            image_size=image_size,
            scheduler_type=scheduler_type,
            num_in_channels=num_in_channels,
            upcast_attention=upcast_attention,
            load_safety_checker=load_safety_checker,
            prediction_type=prediction_type,
            paddle_dtype=paddle_dtype,
        )

        return pipe


def http_file_name(
    url: str,
    *,
    proxies=None,
    headers: Optional[Dict[str, str]] = None,
    timeout=10.0,
    max_retries=0,
):
    """
    Get a remote file name.
    """
    headers = copy.deepcopy(headers) or {}
    r = _request_wrapper(
        method="GET",
        url=url,
        stream=True,
        proxies=proxies,
        headers=headers,
        timeout=timeout,
        max_retries=max_retries,
    )
    hf_raise_for_status(r)
    displayed_name = url.split("/")[-1]
    content_disposition = r.headers.get("Content-Disposition")
    if content_disposition is not None and "filename=" in content_disposition:
        # Means file is on CDN
        displayed_name = content_disposition.split("filename=")[-1]
    return displayed_name
