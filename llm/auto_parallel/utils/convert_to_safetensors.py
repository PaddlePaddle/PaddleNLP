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

import json
import os

import paddle
from safetensors.numpy import save_file as safe_save_file

from paddlenlp.transformers.utils import dtype_byte_size
from paddlenlp.utils.env import SAFE_WEIGHTS_INDEX_NAME


def convert_to_unified_ckpt(path: str, output_dir: str = "./tmp", split_num: int = 1, offload: bool = False):
    """
    Convert a single card checkpoint to the unified format.

    Args:
        path (str): The path to the input checkpoint file.
        output_dir (str, optional): The directory where the converted files will be saved. Defaults to ".".
        split_num (int, optional): The number of shards to split the weights into output_dir. Defaults to 1.
        offload (bool, optional): Whether to offload the weights to CPU memory before saving them. Defaults to False.
    """

    def get_sub_state_dict(sub_keys, state_dict, weight_filename, index_weight_file, total_size):
        """
        Get the sub-state dict and update the index weight file and total size.
        Args:
            sub_keys (list): A list of keys that belong to this sub-state dict.
            state_dict (dict): The original state dict.
            weight_filename (str): The filename of the corresponding weight file.
            index_weight_file (dict): The dictionary containing the mapping from keys to their corresponding weight filenames.
            total_size (int): The total size of the model so far.
        """
        sub_state_dict = {key: state_dict[key].numpy() for key in sub_keys}
        for key in sub_keys:
            index_weight_file[key] = weight_filename
            total_size += state_dict[key].numel().item() * dtype_byte_size(state_dict[key].dtype)
        return sub_state_dict, total_size

    if offload:
        paddle.set_device("cpu")
    state_dict = paddle.load(path)
    all_keys = list(state_dict.keys())
    split_size = len(all_keys) // split_num
    extra_keys = len(all_keys) % split_num
    index_weight_file = {}
    total_size = 0

    os.makedirs(output_dir, exist_ok=True)

    index = 0
    for rank in range(split_num):
        current_size = split_size + (1 if rank < extra_keys else 0)
        sub_keys = all_keys[index : index + current_size]
        index += current_size
        weight_filename = f"model-{rank+1:04d}-of-{split_num:04d}.safetensors"
        sub_state_dict, total_size = get_sub_state_dict(
            sub_keys, state_dict, weight_filename, index_weight_file, total_size
        )
        safe_save_file(sub_state_dict, os.path.join(output_dir, weight_filename))
    with open(os.path.join(output_dir, SAFE_WEIGHTS_INDEX_NAME), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": index_weight_file}, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="The path to the input checkpoint file.")
    parser.add_argument(
        "--output_dir", type=str, default="./tmp", help="The directory where the converted files will be saved."
    )
    parser.add_argument(
        "--split_num", type=int, default=1, help="The number of shards to split the weights into output_dir."
    )
    parser.add_argument(
        "--offload", type=bool, help="Whether to offload the weights to CPU memory before saving them."
    )
    args = parser.parse_args()
    convert_to_unified_ckpt(args.input_path, args.output_dir, args.split_num, args.offload)
