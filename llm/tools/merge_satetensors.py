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


def parse_arguments():
    """
    parse_arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="the path of model weight")
    parser.add_argument(
        "--model_prefix_name", default="model_state", type=str, required=False, help="model prefix name"
    )
    return parser.parse_args()


def merge(chunk_paths, output_file):
    """
    Merge satetensors files to one pdparams file

    Args:
        chunk_paths (list(str)): the list of satetensors file
        output_file (str): the saved path of pdparams file
    """
    dic = {}
    for f in chunk_paths:
        sub_dic = safetensors.safe_open(f, framework="np")
        for k in sub_dic.keys():
            v = sub_dic.get_tensor(k)
            v = paddle.Tensor(v, zero_copy=True)
            dic[k] = v

    paddle.save(dic, output_file)
    print(f"Merged satetensors files and saved to {output_file}")


if __name__ == "__main__":
    """
    Script to merge satetensors files.
    """
    args = parse_arguments()
    output_file = os.path.join(args.model_path, f"{args.model_prefix_name}.pdparams")
    chunk_paths = []
    for f in os.listdir(args.model_path):
        if f.endswith("safetensors"):
            chunk_paths.append(os.path.join(args.model_path, f))
    merge(chunk_paths, output_file)
