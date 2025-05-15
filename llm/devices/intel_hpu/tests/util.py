#!/usr/bin/python3

#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
import subprocess


def get_model_path(model_path_or_name):
    if os.path.exists(model_path_or_name):
        return model_path_or_name
    else:
        data_home = os.environ.get("DATA_HOME", "/data")
        model_name_path = f"{data_home}/{model_path_or_name}"
        if os.path.exists(model_name_path):
            return model_name_path
        else:
            return None


def get_model_path_hf(model_path_or_name):
    if os.path.exists(model_path_or_name):
        return model_path_or_name
    else:
        hf_home = os.environ.get("HF_HOME", f'{os.environ["HOME"]}/.cache/huggingface')
        model_name_path = f"{hf_home}/hub/models--{model_path_or_name.replace('/', '--')}"
        if os.path.exists(model_name_path):
            model_uuid = subprocess.getoutput(f"cat {model_name_path}/refs/main")
            return f"{model_name_path}/snapshots/{model_uuid}"
        else:
            return None


def model_config_add_type(model_name, mt):
    model_path = get_model_path(model_name)
    if model_path is None:
        return
    with open(f"{model_path}/config.json", "r") as f:
        config_dict = json.load(f)
    if "model_type" not in config_dict:
        config_dict["model_type"] = mt
        with open(f"{model_path}/config.json", "w+") as f:
            json.dump(config_dict, f)
