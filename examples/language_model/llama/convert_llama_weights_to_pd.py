# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
import json
import os
import shutil

import numpy as np
import paddle
import torch

"""
Convert weights to paddle, will output `model_state.pdparams`:

    ```
    python ./convert_llama_weights_to_pd.py \
        --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
    ```

Load model:

    ```
    from .tokenizer import LlamaTokenizer
    from .modeling import LlamaForCausalLM

    tokenizer = LlamaTokenizer.from_pretrained("/output/path/llama-7b/")

    model = LlamaForCausalLM.from_pretrained("/output/path/llama-7b/")
    ```

"""

INTERMEDIATE_SIZE_MAP = {
    "7B": 11008,
    "13B": 13824,
    "30B": 17920,
    "65B": 22016,
}
NUM_SHARDS = {
    "7B": 1,
    "13B": 2,
    "30B": 4,
    "65B": 8,
}


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(model_path, input_base_path, model_size):
    assert model_size in INTERMEDIATE_SIZE_MAP
    paddle.set_device("cpu")
    os.makedirs(model_path, exist_ok=True)

    params = read_json(os.path.join(input_base_path, "params.json"))
    num_shards = NUM_SHARDS[model_size]
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads

    # permute for sliced rotary
    def permute(w):
        return w.reshape([n_heads, dim // n_heads // 2, 2, dim]).transpose([0, 2, 1, 3]).reshape([dim, dim])

    # Load weights
    if model_size == "7B":
        # Not shared
        # (The sharded implementation would also work, but this is simpler.)
        loaded = torch.load(os.path.join(input_base_path, "consolidated.00.pth"), map_location="cpu")
        for k, v in loaded.items():
            loaded[k] = paddle.to_tensor(v.numpy())
    else:
        # Sharded
        loaded = [
            torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu")
            for i in range(num_shards)
        ]
        for i in range(num_shards):
            for k, v in loaded[i].items():
                loaded[i][k] = paddle.to_tensor(v.numpy())

    all_state_dict = {}
    for layer_i in range(n_layers):

        if model_size == "7B":
            state_dict = {
                f"llama.layers.{layer_i}.self_attn.q_proj.weight": permute(
                    loaded[f"layers.{layer_i}.attention.wq.weight"]
                ),
                f"llama.layers.{layer_i}.self_attn.k_proj.weight": permute(
                    loaded[f"layers.{layer_i}.attention.wk.weight"]
                ),
                f"llama.layers.{layer_i}.self_attn.v_proj.weight": loaded[f"layers.{layer_i}.attention.wv.weight"],
                f"llama.layers.{layer_i}.self_attn.o_proj.weight": loaded[f"layers.{layer_i}.attention.wo.weight"],
                f"llama.layers.{layer_i}.mlp.gate_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w1.weight"],
                f"llama.layers.{layer_i}.mlp.down_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w2.weight"],
                f"llama.layers.{layer_i}.mlp.up_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w3.weight"],
                f"llama.layers.{layer_i}.input_layernorm.weight": loaded[f"layers.{layer_i}.attention_norm.weight"],
                f"llama.layers.{layer_i}.post_attention_layernorm.weight": loaded[f"layers.{layer_i}.ffn_norm.weight"],
            }
        else:
            # Sharded
            state_dict = {
                f"llama.layers.{layer_i}.input_layernorm.weight": loaded[0][f"layers.{layer_i}.attention_norm.weight"],
                f"llama.layers.{layer_i}.post_attention_layernorm.weight": loaded[0][
                    f"layers.{layer_i}.ffn_norm.weight"
                ],
            }
            state_dict[f"llama.layers.{layer_i}.self_attn.q_proj.weight"] = permute(
                paddle.concat(
                    [
                        loaded[i][f"layers.{layer_i}.attention.wq.weight"].reshape(
                            [n_heads_per_shard, dims_per_head, dim]
                        )
                        for i in range(num_shards)
                    ],
                    axis=0,
                ).reshape([dim, dim])
            )
            state_dict[f"llama.layers.{layer_i}.self_attn.k_proj.weight"] = permute(
                paddle.concat(
                    [
                        loaded[i][f"layers.{layer_i}.attention.wk.weight"].reshape(
                            [n_heads_per_shard, dims_per_head, dim]
                        )
                        for i in range(num_shards)
                    ],
                    axis=0,
                ).reshape([dim, dim])
            )
            state_dict[f"llama.layers.{layer_i}.self_attn.v_proj.weight"] = paddle.concat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wv.weight"].reshape([n_heads_per_shard, dims_per_head, dim])
                    for i in range(num_shards)
                ],
                axis=0,
            ).reshape([dim, dim])

            state_dict[f"llama.layers.{layer_i}.self_attn.o_proj.weight"] = paddle.concat(
                [loaded[i][f"layers.{layer_i}.attention.wo.weight"] for i in range(num_shards)], axis=1
            )
            state_dict[f"llama.layers.{layer_i}.mlp.gate_proj.weight"] = paddle.concat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in range(num_shards)], axis=0
            )
            state_dict[f"llama.layers.{layer_i}.mlp.down_proj.weight"] = paddle.concat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in range(num_shards)], axis=1
            )
            state_dict[f"llama.layers.{layer_i}.mlp.up_proj.weight"] = paddle.concat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w3.weight"] for i in range(num_shards)], axis=0
            )

        all_state_dict.update(state_dict)

    filename = "model_state.pdparams"

    if model_size == "7B":
        state_dict = {
            "llama.embed_tokens.weight": loaded["tok_embeddings.weight"],
            "llama.norm.weight": loaded["norm.weight"],
            "lm_head.weight": loaded["output.weight"],
        }
    else:
        state_dict = {
            "llama.norm.weight": loaded[0]["norm.weight"],
            "llama.embed_tokens.weight": paddle.concat(
                [loaded[i]["tok_embeddings.weight"] for i in range(num_shards)], axis=1
            ),
            "lm_head.weight": paddle.concat([loaded[i]["output.weight"] for i in range(num_shards)], axis=0),
        }

    all_state_dict.update(state_dict)

    for k, v in all_state_dict.items():
        np_value = v.numpy()
        if k.endswith(".weight") and (np_value.ndim == 2) and k != "llama.embed_tokens.weight":
            np_value = np.transpose(np_value)
            v = np_value
        all_state_dict[k] = paddle.to_tensor(v)

    paddle.save(all_state_dict, os.path.join(model_path, filename))

    config_out = {
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "hidden_size": params["dim"],
        "intermediate_size": INTERMEDIATE_SIZE_MAP[model_size],
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": params["n_heads"],
        "num_hidden_layers": params["n_layers"],
        "rms_norm_eps": params["norm_eps"],
        "use_cache": True,
        "vocab_size": 32000,
    }
    write_json(
        config_out,
        os.path.join(model_path, "config.json"),
    )


def write_tokenizer(tokenizer_path, input_tokenizer_path):
    os.makedirs(tokenizer_path, exist_ok=True)
    write_json({}, os.path.join(tokenizer_path, "special_tokens_map.json"))
    write_json(
        {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "cls_token": "<s>",
            "unk_token": "<unk>",
            "add_bos_token": True,
            "add_eos_token": False,
            "tokenizer_class": "LlamaTokenizer",
        },
        os.path.join(tokenizer_path, "tokenizer_config.json"),
    )
    shutil.copyfile(input_tokenizer_path, os.path.join(tokenizer_path, "sentencepiece.bpe.model"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of Llama weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--model_size",
        choices=["7B", "13B", "30B", "65B"],
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write model and tokenizer",
    )
    args = parser.parse_args()
    write_model(
        model_path=os.path.join(args.output_dir, "llama-{}".format(args.model_size).lower()),
        input_base_path=os.path.join(args.input_dir, args.model_size),
        model_size=args.model_size,
    )
    write_tokenizer(
        tokenizer_path=os.path.join(args.output_dir, "llama-{}".format(args.model_size).lower()),
        input_tokenizer_path=os.path.join(args.input_dir, "tokenizer.model"),
    )


if __name__ == "__main__":
    main()
