# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


# You can use it to convert the CKPT of a super large model into a CKPT
# with a small number of layers by modifying _filter_func.
# Note that the converted directory is missing some config files,
# and you need to manually copy.
# At the same time, you need to modify the layers in the config.json file.

import json
import os
from argparse import ArgumentParser
from glob import glob

from safetensors.numpy import load_file, save_file
from tqdm import tqdm


def _filter_func(weight_name: str) -> bool:
    if "layers." in weight_name and int(weight_name.split("layers.")[1].split(".")[0]) > 5:
        return True
    return False


def main(input_path, output_path):
    # Get weight_map
    os.makedirs(output_path, exist_ok=True)
    model_index_file = os.path.join(input_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    # Filter weights
    safetensor_files = list(glob(os.path.join(input_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files):
        current_state_dict = load_file(safetensor_file)

        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if _filter_func(weight_name):
                weight_map.pop(weight_name)
                print("pop: ", weight_name)
            else:
                new_state_dict[weight_name] = weight
                print("save: ", weight_name)

        if len(new_state_dict) > 0:
            file_name = os.path.basename(safetensor_file)
            new_safetensor_file = os.path.join(output_path, file_name)
            save_file(new_state_dict, new_safetensor_file, metadata={"format": "np"})

    # Update model index
    new_model_index_file = os.path.join(output_path, "model.safetensors.index.json")
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_path, args.output_path)
