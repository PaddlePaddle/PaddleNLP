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

from __future__ import annotations

import json
import os


def convert_configs(model_dir: str, output_dir: str | None = None):
    """convert pytorch config.json to model_config.json

    Args:
        model_dir (str): the directory of model-realted files
    """

    # 1. load the config file
    output_dir = output_dir or model_dir
    target_config_file = os.path.join(output_dir, "model_config.json")

    if os.path.exists(target_config_file):
        return

    config_file = os.path.join(model_dir, "config.json")
    assert os.path.exists(config_file), f"<config.json> not found in <{model_dir}> dir"

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 2. transform the config to opt model file
    target_config = {
        "init_args": [
            {
                "intermediate_size": config["ffn_dim"],
                "attention_probs_dropout_prob": config["attention_dropout"],
                "hidden_dropout_prob": config["dropout"],
                "normalize_before": config["do_layer_norm_before"],
                "word_embed_proj_dim": config["word_embed_proj_dim"],
                "num_attention_heads": config["num_attention_heads"],
                "bos_token_id": config["bos_token_id"],
                "hidden_size": config["hidden_size"],
                "eos_token_id": config["eos_token_id"],
                "hidden_act": config["activation_function"],
                "initializer_range": config["init_std"],
                "max_position_embeddings": config["max_position_embeddings"],
                "num_hidden_layers": config["num_hidden_layers"],
                "pad_token_id": config["pad_token_id"],
                "vocab_size": config["vocab_size"],
                "init_class": "OPTModel",
            }
        ],
        "init_class": "OPTForCausalLM",
    }

    with open(target_config_file, "w", encoding="utf-8") as f:
        json.dump(target_config, f)

    print("convert config successfully ...")


def convert_weights(model_dir: str, output_dir: str | None = None):
    # 1. serach the pytorch_model weight files
    files = [
        file_name
        for file_name in os.listdir(model_dir)
        if file_name.startswith("pytorch_model") and file_name.endswith(".bin")
    ]

    # 2. construct name-mapping
    mappings = [
        ["decoder.embed_tokens.weight", "embeddings.word_embeddings.weight"],
        ["decoder.embed_positions.weight", "embeddings.position_embeddings.weight"],
        ["decoder.final_layer_norm.weight", "decoder.final_layer_norm.weight"],
        ["decoder.final_layer_norm.bias", "decoder.final_layer_norm.bias"],
    ]

    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    for layer_index in range(config["num_hidden_layers"]):
        layer_mappings = [
            [
                f"decoder.layers.{layer_index}.self_attn.k_proj.weight",
                f"decoder.layers.{layer_index}.self_attn.k_proj.weight",
                "transpose",
            ],
            [
                f"decoder.layers.{layer_index}.self_attn.k_proj.bias",
                f"decoder.layers.{layer_index}.self_attn.k_proj.bias",
            ],
            [
                f"decoder.layers.{layer_index}.self_attn.v_proj.weight",
                f"decoder.layers.{layer_index}.self_attn.v_proj.weight",
                "transpose",
            ],
            [
                f"decoder.layers.{layer_index}.self_attn.v_proj.bias",
                f"decoder.layers.{layer_index}.self_attn.v_proj.bias",
            ],
            [
                f"decoder.layers.{layer_index}.self_attn.q_proj.weight",
                f"decoder.layers.{layer_index}.self_attn.q_proj.weight",
                "transpose",
            ],
            [
                f"decoder.layers.{layer_index}.self_attn.q_proj.bias",
                f"decoder.layers.{layer_index}.self_attn.q_proj.bias",
            ],
            [
                f"decoder.layers.{layer_index}.self_attn.out_proj.weight",
                f"decoder.layers.{layer_index}.self_attn.out_proj.weight",
                "transpose",
            ],
            [
                f"decoder.layers.{layer_index}.self_attn.out_proj.bias",
                f"decoder.layers.{layer_index}.self_attn.out_proj.bias",
            ],
            [
                f"decoder.layers.{layer_index}.self_attn_layer_norm.weight",
                f"decoder.layers.{layer_index}.norm1.weight",
            ],
            [f"decoder.layers.{layer_index}.self_attn_layer_norm.bias", f"decoder.layers.{layer_index}.norm1.bias"],
            [f"decoder.layers.{layer_index}.fc1.weight", f"decoder.layers.{layer_index}.linear1.weight", "transpose"],
            [f"decoder.layers.{layer_index}.fc1.bias", f"decoder.layers.{layer_index}.linear1.bias"],
            [f"decoder.layers.{layer_index}.fc2.weight", f"decoder.layers.{layer_index}.linear2.weight", "transpose"],
            [f"decoder.layers.{layer_index}.fc2.bias", f"decoder.layers.{layer_index}.linear2.bias"],
            [f"decoder.layers.{layer_index}.final_layer_norm.weight", f"decoder.layers.{layer_index}.norm2.weight"],
            [f"decoder.layers.{layer_index}.final_layer_norm.bias", f"decoder.layers.{layer_index}.norm2.bias"],
        ]
        mappings.extend(layer_mappings)

    # 3. checking the model keys
    import torch
    from tqdm import tqdm

    state_dict = {}
    for file in files:
        file_state_dict = torch.load(file)
        for key in list(file_state_dict.keys()):
            state_dict[key] = file_state_dict.pop(key).numpy()

    for mapping in tqdm(mappings):
        torch_key, paddle_key = mapping[:2]
        assert torch_key in state_dict, f"{torch_key} no in weight file"

    import paddle

    # 4. transform tensor
    from tqdm import tqdm

    for mapping in tqdm(mappings):
        torch_key, paddle_key = mapping[:2]
        value = state_dict.pop(torch_key)
        if len(mapping) == 3:
            value = value.T
        state_dict[paddle_key] = value

    # 5. save the model files
    paddle.save(state_dict, "model_state.pdparams")
    print("convert pytorch model to paddle weight file successfully ...")


if __name__ == "__main__":
    # update your `model_dir` and `output_dir` here to your pytorch model dir
    model_dir = "your pytorch path"
    output_dir = None

    convert_configs(model_dir, output_dir)

    convert_weights(model_dir, output_dir)
