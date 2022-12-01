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

import argparse
import pickle
import re

import paddle


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    return parser.parse_args()


def convert(args):
    paddle.enable_static()
    prog_state = paddle.static.load_program_state(args.param_path)
    new_state = {}
    for k in prog_state:
        if k.endswith("_embedding"):
            prefix = "unified_transformer."
            if k == "word_embedding":
                suffix = "word_embeddings.weight"
            elif k == "pos_embedding":
                suffix = "position_embeddings.weight"
            elif k == "sent_embedding":
                suffix = "token_type_embeddings.weight"
            elif k == "role_embedding":
                suffix = "role_embeddings.weight"
        elif k.startswith("encoder_layer"):
            p = "encoder_layer_(\d+)_([^_]+)_([^_]+)_"
            m = re.match(p, k)
            layer_idx = m.group(1)
            sub_layer = m.group(2)
            prefix = "unified_transformer.encoder.layers." + layer_idx + "."
            if sub_layer == "pre":
                if m.group(3) == "att":
                    if k.endswith("layer_norm_scale"):
                        suffix = "norm1.weight"
                    elif k.endswith("layer_norm_bias"):
                        suffix = "norm1.bias"
                elif m.group(3) == "ffn":
                    if k.endswith("layer_norm_scale"):
                        suffix = "norm2.weight"
                    elif k.endswith("layer_norm_bias"):
                        suffix = "norm2.bias"
            elif sub_layer == "multi":
                prefix += "self_attn."
                m = re.match("encoder_layer_(\d+)_multi_head_att_(\w+)\.(.+)", k)
                if m.group(2) == "query_fc":
                    if m.group(3) == "w_0":
                        suffix = "q_proj.weight"
                    elif m.group(3) == "b_0":
                        suffix = "q_proj.bias"
                elif m.group(2) == "key_fc":
                    if m.group(3) == "w_0":
                        suffix = "k_proj.weight"
                    elif m.group(3) == "b_0":
                        suffix = "k_proj.bias"
                elif m.group(2) == "value_fc":
                    if m.group(3) == "w_0":
                        suffix = "v_proj.weight"
                    elif m.group(3) == "b_0":
                        suffix = "v_proj.bias"
                elif m.group(2) == "output_fc":
                    if m.group(3) == "w_0":
                        suffix = "out_proj.weight"
                    elif m.group(3) == "b_0":
                        suffix = "out_proj.bias"
            elif sub_layer == "ffn":
                if k.endswith("fc_0.w_0"):
                    suffix = "linear1.weight"
                elif k.endswith("fc_0.b_0"):
                    suffix = "linear1.bias"
                elif k.endswith("fc_1.w_0"):
                    suffix = "linear2.weight"
                elif k.endswith("fc_1.b_0"):
                    suffix = "linear2.bias"
        elif k.startswith("post_encoder"):
            prefix = "unified_transformer.encoder."
            if k.endswith("_scale"):
                suffix = "norm.weight"
            elif k.endswith("_bias"):
                suffix = "norm.bias"
        elif k.startswith("mask_lm"):
            prefix = "lm_head."
            if k.endswith("layer_norm_scale"):
                suffix = "layer_norm.weight"
            elif k.endswith("layer_norm_bias"):
                suffix = "layer_norm.bias"
            elif k.endswith("trans_fc.w_0"):
                suffix = "transform.weight"
            elif k.endswith("trans_fc.b_0"):
                suffix = "transform.bias"
            elif k.endswith("out_fc.w_0"):
                suffix = "decoder_weight"
            elif k.endswith("out_fc.b_0"):
                suffix = "decoder_bias"
        new_state[prefix + suffix] = prog_state[k]
    with open(args.save_path, "wb") as f:
        pickle.dump(new_state, f)


if __name__ == "__main__":
    args = setup_args()
    convert(args)
