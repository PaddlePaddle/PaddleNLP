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

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE, hf_cache_home

ppnlp_cache_home = os.path.expanduser(
    os.getenv("PPNLP_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "paddlenlp"))
)

ppdiffusers_default_cache_path = os.path.join(ppnlp_cache_home, "ppdiffusers")
# diffusers_default_cache_path = os.path.join(HUGGINGFACE_HUB_CACHE, "diffusers")
diffusers_default_cache_path = HUGGINGFACE_HUB_CACHE

CONFIG_NAME = "config.json"
TORCH_WEIGHTS_NAME = "diffusion_pytorch_model.bin"
TORCH_SAFETENSORS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
HUGGINGFACE_CO_RESOLVE_ENDPOINT = "https://huggingface.co"
PPDIFFUSERS_CACHE = ppdiffusers_default_cache_path
DIFFUSERS_CACHE = diffusers_default_cache_path
DIFFUSERS_DYNAMIC_MODULE_NAME = "diffusers_modules"
PPDIFFUSERS_DYNAMIC_MODULE_NAME = "ppdiffusers_modules"
HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(hf_cache_home, "modules"))
PPDIFFUSERS_MODULES_CACHE = os.getenv("PPDIFFUSERS_MODULES_CACHE", os.path.join(ppnlp_cache_home, "modules"))

PADDLE_WEIGHTS_NAME = "model_state.pdparams"
FASTDEPLOY_WEIGHTS_NAME = "inference.pdiparams"
FASTDEPLOY_MODEL_NAME = "inference.pdmodel"
WEIGHTS_NAME = PADDLE_WEIGHTS_NAME

TEST_DOWNLOAD_SERVER = "https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/tests"
DOWNLOAD_SERVER = "https://bj.bcebos.com/paddlenlp/models/community"
PPNLP_BOS_RESOLVE_ENDPOINT = os.getenv("PPNLP_ENDPOINT", "https://bj.bcebos.com/paddlenlp")
DEPRECATED_REVISION_ARGS = ["fp16", "non-ema"]
NEG_INF = -1e4

FROM_HF_HUB = os.getenv("FROM_HF_HUB", False)
FROM_DIFFUSERS = os.getenv("FROM_DIFFUSERS", False)
TO_DIFFUSERS = os.getenv("TO_DIFFUSERS", False)
