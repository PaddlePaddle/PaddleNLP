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
