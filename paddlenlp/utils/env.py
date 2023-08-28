# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""
This module is used to store environmental variables in PaddleNLP.
PPNLP_HOME              -->  the root directory for storing PaddleNLP related data. Default to ~/.paddlenlp. Users can change the
├                            default value through the PPNLP_HOME environment variable.
├─ MODEL_HOME              -->  Store model files.
└─ DATA_HOME         -->  Store automatically downloaded datasets.
"""
import os


def _get_user_home():
    return os.path.expanduser("~")


def _get_ppnlp_home():
    if "PPNLP_HOME" in os.environ:
        home_path = os.environ["PPNLP_HOME"]
        if os.path.exists(home_path):
            if os.path.isdir(home_path):
                return home_path
            else:
                raise RuntimeError("The environment variable PPNLP_HOME {} is not a directory.".format(home_path))
        else:
            return home_path
    return os.path.join(_get_user_home(), ".paddlenlp")


def _get_sub_home(directory, parent_home=_get_ppnlp_home()):
    home = os.path.join(parent_home, directory)
    if not os.path.exists(home):
        os.makedirs(home, exist_ok=True)
    return home


def _get_bool_env(env_key: str, default_value: str) -> bool:
    """get boolean environment variable, which can be "true", "True", "1"

    Args:
        env_key (str): key of env variable
    """
    value = os.getenv(env_key, default_value).lower()
    return value in ["true", "1"]


USER_HOME = _get_user_home()
PPNLP_HOME = _get_ppnlp_home()
MODEL_HOME = _get_sub_home("models")
HF_CACHE_HOME = os.environ.get("HUGGINGFACE_HUB_CACHE", MODEL_HOME)
DATA_HOME = _get_sub_home("datasets")
PACKAGE_HOME = _get_sub_home("packages")
DOWNLOAD_SERVER = "http://paddlepaddle.org.cn/paddlehub"
FAILED_STATUS = -1
SUCCESS_STATUS = 0

LEGACY_CONFIG_NAME = "model_config.json"
CONFIG_NAME = "config.json"
TOKENIZER_CONFIG_NAME = "tokenizer_config.json"


LORA_CONFIG_NAME = "lora_config.json"
LORA_WEIGHTS_NAME = "lora_model_state.pdparams"

PREFIX_CONFIG_NAME = "prefix_config.json"
PREFIX_WEIGHTS_NAME = "prefix_model_state.pdparams"

PAST_KEY_VALUES_FILE_NAME = "pre_caches.npy"

PADDLE_WEIGHTS_NAME = "model_state.pdparams"
PADDLE_WEIGHTS_INDEX_NAME = "model_state.pdparams.index.json"

PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
PYTORCH_WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"

SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
