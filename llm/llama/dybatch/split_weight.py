# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import shutil

import paddle

from paddlenlp.transformers import AutoTokenizer

# Define Weight Name List.
LLAMA_COLUMN_SPLIT_WEIGHT_LIST = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
]

LLAMA_ROW_SPLIT_WEIGHT_LIST = ["self_attn.o_proj.weight", "mlp.down_proj.weight"]

LLAMA_NO_SPLIT_WEIGHT_LIST = ["input_layernorm.weight", "post_attention_layernorm.weight"]

LM_HEAD_COLUMN_SPLIT_WEIGHT_LIST = ["lm_head.weight"]

EMBEDDING_ROW_SPLIT_WEIGHT_LIST = ["llama.embed_tokens.weight"]

FINAL_NORM_WEIGHT_LSIT = ["llama.norm.weight"]


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="The directory of model.")
    parser.add_argument("--output_model_dir", required=True, help="The directory of model.")
    parser.add_argument("--nranks", type=int, default="1", help="The number of distributed model num. ")
    return parser.parse_args()


def col_split(weight, nranks):
    return paddle.split(paddle.to_tensor(weight, place=paddle.CPUPlace()), axis=1, num_or_sections=nranks)


def row_split(weight, nranks):
    return paddle.split(paddle.to_tensor(weight, place=paddle.CPUPlace()), axis=0, num_or_sections=nranks)


def split_model_weight(model_dir, nranks, output_model_path):
    model_state_path = os.path.join(model_dir, "model_state.pdparams")
    origin_model = paddle.load(model_state_path, return_numpy=True)
    config_path = os.path.join(model_dir, "config.json")
    config = None
    with open(config_path, "r") as f:
        config = json.load(f)

    for rank_id in range(nranks):
        split_state_dict = dict()
        col_split_lm_head_weight = col_split(origin_model[LM_HEAD_COLUMN_SPLIT_WEIGHT_LIST[0]], nranks)[rank_id]
        row_split_embed_token_weight = row_split(origin_model[EMBEDDING_ROW_SPLIT_WEIGHT_LIST[0]], nranks)[rank_id]
        split_state_dict[LM_HEAD_COLUMN_SPLIT_WEIGHT_LIST[0]] = col_split_lm_head_weight
        split_state_dict[EMBEDDING_ROW_SPLIT_WEIGHT_LIST[0]] = row_split_embed_token_weight

        for layer_id in range(config["num_hidden_layers"]):
            for column_split_weight_name in LLAMA_COLUMN_SPLIT_WEIGHT_LIST:
                full_column_split_weight_name = "llama.layers.{}.".format(layer_id) + column_split_weight_name
                column_split_weight = col_split(origin_model[full_column_split_weight_name], nranks)[rank_id]
                split_state_dict[full_column_split_weight_name] = column_split_weight

            for row_split_weight_name in LLAMA_ROW_SPLIT_WEIGHT_LIST:
                full_row_split_weight_name = "llama.layers.{}.".format(layer_id) + row_split_weight_name
                row_split_weight = row_split(origin_model[full_row_split_weight_name], nranks)[rank_id]
                split_state_dict[full_row_split_weight_name] = row_split_weight

            for no_split_weight_name in LLAMA_NO_SPLIT_WEIGHT_LIST:
                full_no_split_weight_name = "llama.layers.{}.".format(layer_id) + no_split_weight_name
                split_state_dict[full_no_split_weight_name] = paddle.to_tensor(
                    origin_model[full_no_split_weight_name], place=paddle.CPUPlace()
                )

        last_norm_weight_name = FINAL_NORM_WEIGHT_LSIT[0]
        split_state_dict[last_norm_weight_name] = paddle.to_tensor(
            origin_model[last_norm_weight_name], place=paddle.CPUPlace()
        )
        paddle.save(split_state_dict, os.path.join(output_model_path, "model_state.tp0{}.pdparams".format(rank_id)))
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.output_model_dir)


if __name__ == "__main__":
    args = parse_arguments()
    split_model_weight(args.model_dir, args.nranks, args.output_model_dir)
    shutil.copyfile(os.path.join(args.model_dir, "config.json"), os.path.join(args.output_model_dir, "config.json"))
