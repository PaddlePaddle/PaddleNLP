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

import argparse
import json
import os

import paddle
import safetensors
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("merge params")
    parser.add_argument("--param_path", type=str, required=True)
    args = parser.parse_args()
    return args


def merge_safetensors(param_path):
    files = os.listdir(param_path)
    params = {}
    for f in files:
        if f.endswith(".safetensors") and f.startswith("model"):
            params[f] = safetensors.safe_open(os.path.join(param_path, f), framework="np")
    weight_map = json.load(open(os.path.join(param_path, "model.safetensors.index.json"), "r"))["weight_map"]
    model_state = {}
    for k, v in tqdm(weight_map.items()):
        model_state[k] = params[v].get_tensor(k)
    paddle.save(model_state, os.path.join(param_path, "model_state.pdparams"))


def merge_pdparams(param_path):
    files = os.listdir(param_path)
    params = {}
    for f in files:
        if f.endswith(".pdparams") and f != "model_state.pdparams":
            params[f] = paddle.load(os.path.join(param_path, f))
    if os.path.exists(os.path.join(param_path, "model_state.pdparams.index.json")):
        os.system(
            "mv {} {}".format(
                os.path.join(param_path, "model_state.pdparams.index.json"),
                os.path.join(param_path, "pdparams.index.json"),
            )
        )
    weight_map = json.load(open(os.path.join(param_path, "pdparams.index.json"), "r"))["weight_map"]
    model_state = {}
    for k, v in tqdm(weight_map.items()):
        model_state[k] = params[v].get(k)
    paddle.save(model_state, os.path.join(param_path, "model_state.pdparams"))


def main(param_path):
    if os.path.exists(os.path.join(param_path, "model.safetensors.index.json")):
        merge_safetensors(param_path)
    else:
        merge_pdparams(param_path)


if __name__ == "__main__":
    args = parse_args()
    main(args.param_path)
