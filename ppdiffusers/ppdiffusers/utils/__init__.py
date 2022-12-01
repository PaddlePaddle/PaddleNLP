# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from .deprecation_utils import deprecate
from .import_utils import (
    ENV_VARS_TRUE_AND_AUTO_VALUES,
    ENV_VARS_TRUE_VALUES,
    USE_PADDLE,
    DummyObject,
    is_inflect_available,
    is_modelcards_available,
    is_onnx_available,
    is_scipy_available,
    is_paddle_available,
    is_paddlenlp_available,
    is_unidecode_available,
    is_fastdeploy_available,
    requires_backends,
)
from .logging import get_logger
from .outputs import BaseOutput

if is_paddle_available():
    from .testing_utils import floats_tensor, load_image, parse_flag_from_env, slow

logger = get_logger(__name__)

from paddlenlp.utils.env import _get_sub_home, _get_ppnlp_home

ppnlp_cache_home = _get_ppnlp_home()
default_cache_path = _get_sub_home('models')

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model_state.pdparams"
ONNX_WEIGHTS_NAME = "model.onnx"
FASTDEPLOY_WEIGHTS_NAME = "inference.pdiparams"
FASTDEPLOY_MODEL_NAME = "inference.pdmodel"
DOWNLOAD_SERVER = "https://bj.bcebos.com/paddlenlp/models/community"
PPDIFFUSERS_CACHE = default_cache_path
PPDIFFUSERS_DYNAMIC_MODULE_NAME = "ppdiffusers_modules"
PPNLP_MODULES_CACHE = os.getenv("PPNLP_MODULES_CACHE", _get_sub_home("modules"))
