# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from collections import OrderedDict

dont_transpose = [
    "shared.weight",
    "layer_norm.weight",
    ".layer_norm.weight",
    "relative_attention_bias.weight",
    "embed_tokens.weight",
]


def convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path, paddle_dump_path):
    import paddle
    import torch

    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    for k, v in pytorch_state_dict.items():
        transpose = False

        if k[-7:] == ".weight":
            if not any([w in k for w in dont_transpose]):
                if v.ndim == 2:
                    v = v.transpose(0, 1)
                    transpose = True

        print(f"Converting: {k} | is_transpose {transpose}")

        if k != "lm_head.weight":
            k = "mt5." + k
        paddle_state_dict[k] = v.data.numpy()

    paddle.save(paddle_state_dict, paddle_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch_checkpoint_path",
        default="google/mt5-small/pytorch_model.bin",
        type=str,
        required=False,
        help="Path to the Pytorch checkpoint path.",
    )
    parser.add_argument(
        "--paddle_dump_path",
        default="paddle/mt5-small/model_state.pdparams",
        type=str,
        required=False,
        help="Path to the output Paddle model.",
    )
    args = parser.parse_args()
    convert_pytorch_checkpoint_to_paddle(args.pytorch_checkpoint_path, args.paddle_dump_path)
