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

huggingface_to_paddle = {
    ".attn.": ".",
    "intermediate.dense": "ffn",
    "output.dense": "ffn_output",
    ".output.LayerNorm.": ".layer_norm.",
    ".LayerNorm.": ".layer_norm.",
    "lm_head.decoder.bias": "lm_head.decoder_bias",
}

skip_weights = ["lm_head.decoder.weight", "lm_head.bias"]
dont_transpose = [
    "_embeddings.weight",
    ".LayerNorm.weight",
    ".layer_norm.weight",
    "relative_attention_bias.weight",
]


def convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path,
                                         paddle_dump_path):
    import torch
    import paddle

    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    for k, v in pytorch_state_dict.items():
        transpose = False
        if k in skip_weights:
            continue
        if k[-7:] == ".weight":
            if not any([w in k for w in dont_transpose]):
                if v.ndim == 2:
                    v = v.transpose(0, 1)
                    transpose = True
        oldk = k
        for huggingface_name, paddle_name in huggingface_to_paddle.items():
            k = k.replace(huggingface_name, paddle_name)

        print(f"Converting: {oldk} => {k} | is_transpose {transpose}")
        paddle_state_dict[k] = v.data.numpy()

    paddle.save(paddle_state_dict, paddle_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch_checkpoint_path",
        default="weights/hg/mpnet-base/pytorch_model.bin",
        type=str,
        required=False,
        help="Path to the Pytorch checkpoint path.",
    )
    parser.add_argument(
        "--paddle_dump_path",
        default="weights/pd/mpnet-base/model_state.pdparams",
        type=str,
        required=False,
        help="Path to the output Paddle model.",
    )
    args = parser.parse_args()
    convert_pytorch_checkpoint_to_paddle(args.pytorch_checkpoint_path,
                                         args.paddle_dump_path)
