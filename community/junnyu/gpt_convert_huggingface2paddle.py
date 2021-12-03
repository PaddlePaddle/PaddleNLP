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
    "transformer.wte.weight": "gpt.embeddings.word_embeddings.weight",
    "transformer.wpe.weight": "gpt.embeddings.position_embeddings.weight",
    "transformer.h.": "gpt.decoder.layers.",
    ".attn.c_proj.": ".self_attn.out_proj.",
    ".ln_1.": ".norm1.",
    ".mlp.c_fc.": ".linear1.",
    ".mlp.c_proj.": ".linear2.",
    ".ln_2.": ".norm2.",
    "transformer.ln_f.": "gpt.decoder.norm.",
    "lm_head.weight": "lm_head.decoder_weight"
}

skip_weights = [".attn.bias", "lm_head.weight"]
dont_transpose = [
    ".wte.weight", ".wpe.weight", ".ln_", ".mlp.c_proj.", ".mlp.c_fc.",
    ".attn.c_proj.", "lm_head.weight"
]


# 注意，huggingface使用的Conv1D的weight和paddle.nn.Linear中的weight形状一致，因此不需要转置。
# 如果使用了torch.nn.Linear那么就需要转置了！
def convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path,
                                         paddle_dump_path):
    import torch
    import paddle
    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    for k, v in pytorch_state_dict.items():
        is_transpose = False
        if k in skip_weights:
            continue
        # c_attn
        if ".attn.c_attn." in k:
            query_value_key = v.chunk(chunks=3, dim=-1)
            for cross_value, new_name in zip(query_value_key, [
                    ".self_attn.q_proj.", ".self_attn.k_proj.",
                    ".self_attn.v_proj."
            ]):
                oldk = k
                newk = k.replace("transformer.h.",
                                 "gpt.decoder.layers.").replace(".attn.c_attn.",
                                                                new_name)
                paddle_state_dict[newk] = cross_value.data.numpy().astype(
                    "float32")
                print(
                    f"Converting: {oldk} => {newk} | is_transpose {is_transpose}"
                )
            continue

        if k[-7:] == ".weight":
            if not any([w in k for w in dont_transpose]):
                if v.ndim == 2:
                    v = v.transpose(0, 1)
                    is_transpose = True
        oldk = k
        for huggingface_name, paddle_name in huggingface_to_paddle.items():
            k = k.replace(huggingface_name, paddle_name)

        print(f"Converting: {oldk} => {k} | is_transpose {is_transpose}")
        paddle_state_dict[k] = v.data.numpy().astype("float32")

    paddle.save(paddle_state_dict, paddle_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch_checkpoint_path",
        default=r"community\junnyu\microsoft-DialoGPT-small\pytorch_model.bin",
        type=str,
        required=False,
        help="Path to the Pytorch checkpoint path.")
    parser.add_argument(
        "--paddle_dump_path",
        default=r"community\junnyu\microsoft-DialoGPT-small\model_state.pdparams",
        type=str,
        required=False,
        help="Path to the output Paddle model.")
    args = parser.parse_args()
    convert_pytorch_checkpoint_to_paddle(args.pytorch_checkpoint_path,
                                         args.paddle_dump_path)
