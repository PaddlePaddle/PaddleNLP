# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from collections import OrderedDict
import argparse
import json
import os

dont_transpose = [
    "shared.weight", "layer_norm.weight", ".layer_norm.weight",
    "relative_attention_bias.weight", "embed_tokens.weight"
]


def convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path,
                                         paddle_dump_path):
    import torch
    import paddle
    import numpy as np
    parameter_count = 0
    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    for k, v in pytorch_state_dict.items():
        transpose = False

        if k[-7:] == ".weight":
            if not any([w in k for w in dont_transpose]):
                if v.ndim == 2:
                    v = v.transpose(0, 1)
                    transpose = True

        print(f"Converting: {k} {v.size()} | is_transpose {transpose}")

        if k != "lm_head.weight":
            k = "t5." + k
        paddle_state_dict[k] = v.data.numpy()

        parameter_count += np.prod(v.size())

    paddle.save(paddle_state_dict, paddle_dump_path)
    print(f"Total parameter number: {parameter_count}")


def convert_pytorch_config_to_paddle(pytorch_config, paddle_config):
    model_config = json.load(open(pytorch_config))

    to_pop_list = [
        '_name_or_path', 'architectures', 'decoder_start_token_id', 'use_cache',
        'torch_dtype', 'transformers_version', 'output_past',
        'is_encoder_decoder', 'model_type', 'gradient_checkpointing',
        'max_length'
    ]

    for to_pop in to_pop_list:
        if to_pop not in model_config:
            continue
        model_config.pop(to_pop)

    json.dump(model_config, open(paddle_config, 'w'), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch_checkpoint_path",
        default="google/t5-large",
        type=str,
        required=False,
        help="Path to the Pytorch checkpoint path.", )
    parser.add_argument(
        "--paddle_dump_path",
        default="paddle/t5-large",
        type=str,
        required=False,
        help="Path to the output Paddle model.", )
    args = parser.parse_args()
    convert_pytorch_config_to_paddle(
        os.path.join(args.pytorch_checkpoint_path, "config.json"),
        os.path.join(args.paddle_dump_path, "model_config.json"), )
    convert_pytorch_checkpoint_to_paddle(
        os.path.join(args.pytorch_checkpoint_path, "pytorch_model.bin"),
        os.path.join(args.paddle_dump_path, "model_state.pdparams"), )
