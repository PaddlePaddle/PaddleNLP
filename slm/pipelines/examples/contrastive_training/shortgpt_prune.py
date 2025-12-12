# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import math
import os
import re
from collections import OrderedDict
from typing import List

import paddle
from datasets import load_dataset
from paddle.io import DataLoader
from tqdm import tqdm

from paddlenlp.transformers import AutoModel, AutoTokenizer, NVEncodeModel


# =====================================================================================
# 1. block_influence
# =====================================================================================
def block_influence(
    input_hidden_state: paddle.Tensor,
    output_hidden_state: paddle.Tensor,
    angular: bool = False,
) -> paddle.Tensor:
    """
    Calculates block influence between input and output hidden states.
    """
    _, _, d = input_hidden_state.shape
    input_hidden_state = paddle.reshape(input_hidden_state, [-1, d])
    output_hidden_state = paddle.reshape(output_hidden_state, [-1, d])

    norm_input = paddle.norm(input_hidden_state, p=2, axis=-1, keepdim=True)
    norm_output = paddle.norm(output_hidden_state, p=2, axis=-1, keepdim=True)

    sim = paddle.matmul(input_hidden_state, output_hidden_state, transpose_y=True) / (norm_input * norm_output)
    sim = paddle.diag(sim).astype("float32").nan_to_num(nan=0.5)

    if angular:
        return paddle.acos(sim) / math.pi
    return 1 - sim


# =====================================================================================
# 2. ShortGPT
# =====================================================================================
class ShortGPT:
    """
    A class to evaluate layer importance in LLMs using PaddlePaddle.
    """

    def __init__(self, model_name: str, layers_path: str):
        print(f"Loading tokenizer for '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model '{model_name}' with PaddlePaddle backend...")
        if "NV-Embed" in model_name:
            self.model = NVEncodeModel.from_pretrained(
                model_name, tokenizer_path=model_name, query_instruction="", document_instruction=""
            )
        else:
            self.model = AutoModel.from_pretrained(model_name, dtype=paddle.float16)

        import sys
        from io import StringIO

        buffer = StringIO()
        sys.stdout = buffer
        print(self.model)
        sys.stdout = sys.__stdout__

        model_str = buffer.getvalue()
        print("\n=== Model structure (first 5 lines) ===")
        print("\n".join(model_str.splitlines()[:5]))

        self.model.eval()
        print("Model loaded successfully for importance evaluation.")

        try:
            path_parts = layers_path.split(".")  # e.g., 'llama.layers' -> ['llama', 'layers']

            self.base_model_for_call = self.model
            # 遍历路径中除了最后 'layers' 之外的部分 (e.g., 'llama')
            for part in path_parts[:-1]:
                self.base_model_for_call = getattr(self.base_model_for_call, part)

            # 从基础模型中获取 'layers' 列表
            self.layers = getattr(self.base_model_for_call, path_parts[-1])
            print(f"Successfully located base model for evaluation call: {type(self.base_model_for_call)}")
            print(f"Successfully located {len(self.layers)} layers.")

        except AttributeError:
            raise AttributeError(f"Could not find layers at path '{layers_path}' in the model architecture.")

        self.importances = [0.0 for _ in self.layers]

    def compute_bi(self, hiddens: List[paddle.Tensor]):
        """
        Computes and accumulates block influence scores from hidden states.
        """
        n = 1
        for i in range(len(hiddens) - n):
            layer_index = i
            if layer_index < len(self.importances):
                in_hidden = hiddens[i]
                out_hidden = hiddens[i + n]
                self.importances[layer_index] += block_influence(in_hidden, out_hidden).sum().item()

    @paddle.no_grad()
    def eval_importance(self, prompts: List[str], model_name: str, stride: int = 256):
        """
        Evaluates the importance of model layers on given prompts.
        """
        prompt_tokens = self.tokenizer(prompts, padding=True, return_attention_mask=True, return_tensors="pd")
        input_ids = prompt_tokens.input_ids
        attn_mask = prompt_tokens.attention_mask

        max_prompt_len = max(len(t) for t in input_ids)

        for start in range(0, max_prompt_len, stride):
            seq_ids = (attn_mask.sum(axis=-1) > start).nonzero().squeeze()
            seq_ids = seq_ids.unsqueeze(0) if seq_ids.ndim == 0 else seq_ids

            if seq_ids.shape[0] == 0:
                continue

            inputs = input_ids[seq_ids, start : start + stride]
            attn = attn_mask[seq_ids, start : start + stride]

            if "NV-Embed" in model_name:
                outputs = self.base_model_for_call.m_forward(
                    input_ids=inputs, attention_mask=attn, output_hidden_states=True, return_dict=True
                )
            else:
                outputs = self.base_model_for_call(
                    input_ids=inputs, attention_mask=attn, output_hidden_states=True, return_dict=True
                )

            if outputs.hidden_states:
                self.compute_bi(outputs.hidden_states)


def load_model_weights(model_folder_path: str) -> OrderedDict:
    print(f"Attempting to load model weights from FOLDER: '{model_folder_path}'...")

    # 1. Ensure the path is a valid directory
    if not os.path.isdir(model_folder_path):
        raise NotADirectoryError(f"The provided path is not a valid directory: '{model_folder_path}'")

    state_dict = OrderedDict()
    index_path = os.path.join(model_folder_path, "model_state.pdparams.index.json")

    # 2. Check for the presence of a sharded index file
    if os.path.isfile(index_path):
        # Case A: Sharded model format detected (index file found)
        print("Sharded model format detected (index file found).")
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        shard_files = sorted(list(set(index_data["weight_map"].values())))
        print(f"Found {len(shard_files)} shard(s).")

        for shard_file in shard_files:
            shard_path = os.path.join(model_folder_path, shard_file)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"Shard file '{shard_file}' listed in index not found at '{shard_path}'")

            print(f"  > Loading shard: {shard_file}")
            shard_state_dict = paddle.load(shard_path, return_numpy=True)
            state_dict.update(shard_state_dict)
            del shard_state_dict
        print("All weight shards loaded successfully.")

    else:
        # Case B: No index file found; look for a single .pdparams file
        print("No index file found. Searching for a single .pdparams file inside the folder...")
        pdparams_files = [f for f in os.listdir(model_folder_path) if f.endswith(".pdparams")]

        if len(pdparams_files) == 1:
            # Found exactly one .pdparams file
            single_file_path = os.path.join(model_folder_path, pdparams_files[0])
            print(f"  > Loading single parameters file: {pdparams_files[0]}")
            state_dict = paddle.load(single_file_path, return_numpy=True)
            print("Single weight file loaded successfully.")
        elif len(pdparams_files) > 1:
            raise ValueError(
                f"Ambiguous model files. Multiple .pdparams files found in '{model_folder_path}' "
                "but no 'model_state.pdparams.index.json' to specify order."
            )
        else:  # len(pdparams_files) == 0
            raise FileNotFoundError(f"No .pdparams files found in the directory '{model_folder_path}'.")

    return state_dict


# =====================================================================================
# 3. Prune and Save
# =====================================================================================
def prune_and_save_model_in_memory(
    model,
    tokenizer,
    new_model_path,
    layers_to_delete,
    layers_path_str,
):
    """
    Prunes and saves a model directly from the in-memory model object.
    """
    print("=" * 50)
    print("PART 2: Starting In-Memory Model Pruning and Saving")
    print("=" * 50)
    os.makedirs(new_model_path, exist_ok=True)

    # Step 1: Get state_dict directly from the in-memory model
    print("Getting state_dict directly from the in-memory model...")
    state_dict = model.state_dict()

    # Step 2: Iterate, filter, and rename weights
    print("Processing weights: removing specified layers and re-indexing...")
    escaped_layers_path = layers_path_str.replace(".", r"\.")
    layer_pattern = re.compile(rf"^{escaped_layers_path}\.(\d+)\.")
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        match = layer_pattern.match(key)
        if not match:
            new_state_dict[key] = value
            continue

        layer_idx = int(match.group(1))
        if layer_idx in layers_to_delete:
            continue

        num_layers_deleted_before = sum(1 for deleted_idx in layers_to_delete if deleted_idx < layer_idx)
        new_layer_idx = layer_idx - num_layers_deleted_before
        old_prefix = f"{layers_path_str}.{layer_idx}."
        new_prefix = f"{layers_path_str}.{new_layer_idx}."
        new_key = key.replace(old_prefix, new_prefix, 1)
        new_state_dict[new_key] = value

    print(f"Processing complete. Removed {len(layers_to_delete)} layer(s): {sorted(list(layers_to_delete))}.")

    # Step 3: Get and modify the configuration from the model object
    print("Updating configuration file...")
    config = model.config.to_dict()

    # Fix: Convert non-serializable paddle data types to strings
    for key, value in config.items():
        if type(value).__name__ == "DataType":
            config[key] = str(value).split(".")[-1]

    if "num_hidden_layers" in config:
        original_num_layers = config["num_hidden_layers"]
        new_num_layers = original_num_layers - len(layers_to_delete)
        config["num_hidden_layers"] = new_num_layers
        print(f"  - Number of layers changed from {original_num_layers} to {new_num_layers}.")

    new_config_path = os.path.join(new_model_path, "config.json")
    with open(new_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"New config saved to '{new_config_path}'.")

    # Step 4: Save the new weights and tokenizer
    print("Saving pruned weights...")
    new_weights_path = os.path.join(new_model_path, "model_state.pdparams")
    paddle.save(new_state_dict, new_weights_path)
    print(f"Pruned weights saved to '{new_weights_path}'.")

    print("Saving tokenizer files...")
    tokenizer.save_pretrained(new_model_path)
    print(f"Tokenizer files saved to '{new_model_path}'.")

    print("\n🎉 Pruning process completed successfully!")
    print(f"Pruned model has been saved to '{new_model_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate layer importance, prune, and save a new PaddlePaddle model."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path or HuggingFace name of the source PaddlePaddle model.",
    )
    parser.add_argument(
        "--output_model_path", type=str, required=True, help="Path to save the new, pruned model directory."
    )
    parser.add_argument(
        "--layers_path", type=str, required=True, help="Dot-separated path to the layers list (e.g., 'llama.layers')."
    )
    parser.add_argument(
        "--n_prune_layers", type=int, required=True, help="The number of layers to identify and prune."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="emozilla/pg19",
        help="Name of the Hugging Face dataset for calibration. Default: 'emozilla/pg19'.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="validation",
        help="The split of the dataset to use. Default: 'validation'.",
    )
    args = parser.parse_args()

    # --- PART 1: Calculate Layer Importance ---
    print("=" * 50)
    print("PART 1: Calculating Layer Importance")
    print("=" * 50)
    print(f"Loading '{args.dataset_split}' split from '{args.dataset_name}' dataset for calibration...")
    try:
        data = load_dataset(args.dataset_name, split=args.dataset_split)
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        print(
            "Please ensure the dataset name and split are correct and you have internet access for Hugging Face datasets."
        )
        return

    dataloader = DataLoader(data, batch_size=1, shuffle=False)

    short_model = ShortGPT(model_name=args.model_name_or_path, layers_path=args.layers_path)

    for batch in tqdm(dataloader, desc="Evaluating Layer Importance"):
        if "text" not in batch:
            raise ValueError("Dataset must contain a 'text' column.")
        prompts = batch["text"]
        short_model.eval_importance(prompts=prompts, model_name=args.model_name_or_path, stride=256)

    prune_order = sorted(range(len(short_model.importances)), key=lambda i: short_model.importances[i])
    layers_to_delete = set(prune_order[: args.n_prune_layers])

    print("\n--- Importance Calculation Complete ---")
    print(f"Calculated importances: {[f'{v:.2f}' for v in short_model.importances]}")
    print(f"Pruning order (least to most important): {prune_order}")
    print(f"Will delete the {args.n_prune_layers} least important layers: {sorted(list(layers_to_delete))}")

    # --- PART 2: Perform In-Memory Pruning and Saving ---
    prune_and_save_model_in_memory(
        model=short_model.model,
        tokenizer=short_model.tokenizer,
        new_model_path=args.output_model_path,
        layers_to_delete=layers_to_delete,
        layers_path_str=args.layers_path,
    )


if __name__ == "__main__":
    main()
