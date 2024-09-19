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
import os

import paddle

from paddlenlp.transformers import AutoConfig, AutoModelForCausalLM
from paddlenlp.transformers.model_utils import load_tp_checkpoint


def parse_arguments():
    """
    parse_arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gqa_model_path", type=str, required=True, help="the dir of gqa_model weight")
    parser.add_argument("--mha_model_path", type=str, required=True, help="the saved dir of mha_model weight")
    parser.add_argument(
        "--model_prefix_name", default="model_state", type=str, required=False, help="model prefix name"
    )
    return parser.parse_args()


def convert(gqa_model_path, mha_model_path, config_path):
    """
    Convert model from gqa to mha

    Args:
        gqa_model_path (str): the path of gqa_model weight
        mha_model_path (str): the saved path of mha_model weight
        config_path (str): the path of model's config
    """
    config = AutoConfig.from_pretrained(gqa_model_path)

    model = AutoModelForCausalLM.from_pretrained(gqa_model_path)
    model_state = load_tp_checkpoint(gqa_model_path, model, config)

    model_type = config["model_type"]
    hidden_size = config["hidden_size"]
    num_head = config["num_attention_heads"]
    num_key_value_heads = config["num_key_value_heads"]
    dim_head = hidden_size // num_head
    num_layers = config["num_hidden_layers"]
    num_gqa_partitions = num_head // num_key_value_heads

    for i in range(num_layers):
        print(f"num_layers: {i}")
        # qkv weight [hidden_size, (num_head + 2 * num_key_value_heads) * dim_head]
        q_weight = model_state[f"{model_type}.layers.{i}.self_attn.q_proj.weight"]
        k_weight = model_state[f"{model_type}.layers.{i}.self_attn.k_proj.weight"]
        v_weight = model_state[f"{model_type}.layers.{i}.self_attn.v_proj.weight"]
        print(f"q_weight.shape: {q_weight.shape}")
        print(f"k_weight.shape: {k_weight.shape}")
        print(f"k_weight.shape: {v_weight.shape}")

        k_weight = k_weight.reshape([hidden_size, num_key_value_heads, dim_head])
        v_weight = v_weight.reshape([hidden_size, num_key_value_heads, dim_head])
        print(f"(reshape) k_weight.shape: {k_weight.shape}")
        print(f"(reshape) v_weight.shape: {v_weight.shape}")

        kk_weight = paddle.reshape(
            paddle.stack([k_weight] * num_gqa_partitions, axis=2), [hidden_size, num_head, dim_head]
        )
        vv_weight = paddle.reshape(
            paddle.stack([v_weight] * num_gqa_partitions, axis=2), [hidden_size, num_head, dim_head]
        )
        print(f"(extend) k_weight.shape: {kk_weight.shape}")
        print(f"(extend) v_weight.shape: {vv_weight.shape}")

        new_k_weight = kk_weight.reshape([hidden_size, num_head * dim_head])
        new_v_weight = vv_weight.reshape([hidden_size, num_head * dim_head])
        print(f"new_k_weight.shape: {new_k_weight.shape}")
        print(f"new_v_weight.shape: {new_v_weight.shape}")

        model_state[f"{model_type}.layers.{i}.self_attn.k_proj.weight"] = new_k_weight
        model_state[f"{model_type}.layers.{i}.self_attn.v_proj.weight"] = new_v_weight

        if (
            f"{model_type}.layers.{i}.self_attn.q_proj.bias" in model_state
            and f"{model_type}.layers.{i}.self_attn.k_proj.bias" in model_state
            and f"{model_type}.layers.{i}.self_attn.v_proj.bias" in model_state
        ):
            print("bias")

            q_bias = model_state[f"{model_type}.layers.{i}.self_attn.q_proj.bias"]
            k_bias = model_state[f"{model_type}.layers.{i}.self_attn.k_proj.bias"]
            v_bias = model_state[f"{model_type}.layers.{i}.self_attn.v_proj.bias"]
            print(f"q_bias.shape: {q_bias.shape}")
            print(f"k_bias.shape: {k_bias.shape}")
            print(f"v_bias.shape: {v_bias.shape}")

            k_bias = k_bias.reshape([num_key_value_heads, dim_head])
            v_bias = v_bias.reshape([num_key_value_heads, dim_head])
            print(f"(reshape) k_bias.shape: {k_bias.shape}")
            print(f"(reshape) v_bias.shape: {v_bias.shape}")

            kk_bias = paddle.reshape(paddle.stack([k_bias] * num_gqa_partitions, axis=1), [num_head, dim_head])
            vv_bias = paddle.reshape(paddle.stack([v_bias] * num_gqa_partitions, axis=1), [num_head, dim_head])
            print(f"(extend) k_bias.shape: {kk_bias.shape}")
            print(f"(extend) v_bias.shape: {vv_bias.shape}")

            new_k_bias = kk_bias.reshape([num_head * dim_head])
            new_v_bias = vv_bias.reshape([num_head * dim_head])
            print(f"new_k_bias.shape: {new_k_bias.shape}")
            print(f"new_v_bias.shape: {new_v_bias.shape}")

            model_state[f"{model_type}.layers.{i}.self_attn.k_proj.bias"] = new_k_bias
            model_state[f"{model_type}.layers.{i}.self_attn.v_proj.bias"] = new_v_bias

    paddle.save(model_state, mha_model_path)


if __name__ == "__main__":
    """
    Script to convert model from gqa to mha.
    """
    args = parse_arguments()
    config_path = os.path.join(args.gqa_model_path, "config.json")
    mha_model_path = os.path.join(args.mha_model_path, f"{args.model_prefix_name}.pdparams")

    assert os.path.exists(config_path), "config.json is not found in {}".format(args.gqa_model_path)
    assert os.path.exists(args.gqa_model_path), "{} is not found".format(args.gqa_model_path)
    convert(args.gqa_model_path, mha_model_path, config_path)
