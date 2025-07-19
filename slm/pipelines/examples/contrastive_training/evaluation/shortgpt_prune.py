import argparse
import json
import math
import os
import re
from collections import OrderedDict
from shutil import copyfile
from typing import List, Optional

import numpy as np
import paddle
from datasets import load_dataset
from paddle.io import DataLoader
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# =====================================================================================
# 1. 块影响计算函数
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
    sim = paddle.diag(sim).astype('float32').nan_to_num(nan=0.5)

    if angular:
        return paddle.acos(sim) / math.pi
    return 1 - sim

# =====================================================================================
# 2. ShortGPT 核心类
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=paddle.float16
        )
        
        self.model.eval()
        print("Model loaded successfully for importance evaluation.")

        try:
            modules = layers_path.split(".")
            mod = self.model
            for m in modules:
                mod = getattr(mod, m)
            self.layers = mod
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
                out_hidden = hiddens[i+n]
                self.importances[layer_index] += block_influence(
                    in_hidden,
                    out_hidden
                ).sum().item()

    @paddle.no_grad()
    def eval_importance(self, prompts: List[str], stride: int = 256):
        """
        Evaluates the importance of model layers on given prompts.
        """
        prompt_tokens = self.tokenizer(
            prompts, padding=True, return_attention_mask=True, return_tensors='pd'
        )
        input_ids = prompt_tokens.input_ids
        attn_mask = prompt_tokens.attention_mask

        max_prompt_len = max(len(t) for t in input_ids)

        for start in range(0, max_prompt_len, stride):
            seq_ids = (attn_mask.sum(axis=-1) > start).nonzero().squeeze()
            seq_ids = seq_ids.unsqueeze(0) if seq_ids.ndim == 0 else seq_ids
            
            if seq_ids.shape[0] == 0:
                continue

            inputs = input_ids[seq_ids, start:start+stride]
            attn = attn_mask[seq_ids, start:start+stride]
            
            outputs = self.model(
                input_ids=inputs,
                attention_mask=attn,
                output_hidden_states=True,
                return_dict=True
            )
            
            if outputs.hidden_states:
                self.compute_bi(outputs.hidden_states)

def load_model_weights(model_folder_path: str) -> OrderedDict:
    """
    从一个指定的【文件夹】中加载模型权重。
    - 首先检查是否存在分片索引文件 (`model_state.pdparams.index.json`)。
    - 如果没有，则在该文件夹中查找单个 `.pdparams` 文件。

    Args:
        model_folder_path (str): 模型权重所在的【文件夹】路径。

    Returns:
        OrderedDict: 加载后的模型状态字典。
        
    Raises:
        NotADirectoryError: 如果提供的路径不是一个有效的文件夹。
        FileNotFoundError: 如果文件夹中找不到任何可加载的权重文件。
        ValueError: 如果文件夹中存在多个 .pdparams 文件但没有索引文件。
    """
    print(f"Attempting to load model weights from FOLDER: '{model_folder_path}'...")
    
    # 1. 强制路径必须是一个文件夹
    if not os.path.isdir(model_folder_path):
        raise NotADirectoryError(f"The provided path is not a valid directory: '{model_folder_path}'")

    state_dict = OrderedDict()
    index_path = os.path.join(model_folder_path, "model_state.pdparams.index.json")

    # 2. 检查文件夹内是否有分片索引
    if os.path.isfile(index_path):
        # 场景A: 文件夹中包含分片索引，按分片加载
        print("Sharded model format detected (index file found).")
        with open(index_path, 'r', encoding='utf-8') as f:
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
        # 场景B: 文件夹中无索引，查找单个 .pdparams 文件
        print("No index file found. Searching for a single .pdparams file inside the folder...")
        pdparams_files = [f for f in os.listdir(model_folder_path) if f.endswith(".pdparams")]
        
        if len(pdparams_files) == 1:
            # 找到了唯一的一个 .pdparams 文件
            single_file_path = os.path.join(model_folder_path, pdparams_files[0])
            print(f"  > Loading single parameters file: {pdparams_files[0]}")
            state_dict = paddle.load(single_file_path, return_numpy=True)
            print("Single weight file loaded successfully.")
        elif len(pdparams_files) > 1:
            # 找到了多个，情况不明，抛出错误
            raise ValueError(
                f"Ambiguous model files. Multiple .pdparams files found in '{model_folder_path}' "
                "but no 'model_state.pdparams.index.json' to specify order."
            )
        else: # len(pdparams_files) == 0
            # 一个 .pdparams 文件都找不到
            raise FileNotFoundError(
                f"No .pdparams files found in the directory '{model_folder_path}'."
            )

    return state_dict


# =====================================================================================
# 3. 剪枝与保存
# =====================================================================================
def prune_and_save_model_in_memory(
    model,
    tokenizer,
    new_model_path,
    layers_to_delete,
    layers_path_str,
):
    """
    直接从内存中的模型对象进行剪枝并保存。

    Args:
        model (PretrainedModel): 已加载到内存中的完整模型对象。
        tokenizer (PretrainedTokenizer): 对应的分词器对象。
        new_model_path (str): 用于保存剪枝后新模型的路径。
        layers_to_delete (Set[int]): 要移除的层索引集合。
        layers_path_str (str): 指向层的点分隔路径字符串 (e.g., "model.layers")。
    """
    print("\n" + "="*50)
    print("PART 2: Starting In-Memory Model Pruning and Saving")
    print("="*50)
    os.makedirs(new_model_path, exist_ok=True)

    # --- 步骤 1: 直接从内存中的模型获取 state_dict ---
    print("Getting state_dict directly from the in-memory model...")
    state_dict = model.state_dict()

    # --- 步骤 2: 遍历、筛选和重命名权重 (与之前逻辑相同) ---
    print("Processing weights: removing specified layers and re-indexing...")
    escaped_layers_path = layers_path_str.replace('.', r'\.')
    layer_pattern = re.compile(rf"^{escaped_layers_path}\.(\d+)\.")
    new_state_dict = OrderedDict()
    layers_deleted_count = set()

    for key, value in state_dict.items():
        match = layer_pattern.match(key)
        if not match:
            new_state_dict[key] = value
            continue
        layer_idx = int(match.group(1))
        if layer_idx in layers_to_delete:
            layers_deleted_count.add(layer_idx)
            continue
        num_layers_deleted_before = sum(1 for deleted_idx in layers_to_delete if deleted_idx < layer_idx)
        new_layer_idx = layer_idx - num_layers_deleted_before
        old_prefix = f"{layers_path_str}.{layer_idx}."
        new_prefix = f"{layers_path_str}.{new_layer_idx}."
        new_key = key.replace(old_prefix, new_prefix, 1)
        new_state_dict[new_key] = value

    print(f"Processing complete. Removed {len(layers_deleted_count)} layer(s): {sorted(list(layers_deleted_count))}.")

    # --- 步骤 3: 直接从模型对象获取并修改配置 ---
    print("Updating configuration file...")
    config = model.config.to_dict()
    
    for key, value in config.items():
        # 不再使用 isinstance，而是直接判断对象的类名是否为 'DataType'
        if type(value).__name__ == 'DataType':
            # 将 paddle.float16 这样的对象，转换为 "float16" 这样的字符串
            config[key] = str(value).split('.')[-1]
            
    if "num_hidden_layers" in config:
        original_num_layers = config["num_hidden_layers"]
        new_num_layers = original_num_layers - len(layers_to_delete)
        config["num_hidden_layers"] = new_num_layers
        print(f"  - Number of layers changed from {original_num_layers} to {new_num_layers}.")

    new_config_path = os.path.join(new_model_path, "config.json")
    with open(new_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    print(f"New config saved to '{new_config_path}'.")

    # --- 步骤 4: 保存新的权重文件和 Tokenizer ---
    print("Saving pruned weight file...")
    new_weights_path = os.path.join(new_model_path, "model_state.pdparams")
    paddle.save(new_state_dict, new_weights_path)
    print(f"New weights saved to '{new_weights_path}'.")

    print("Saving tokenizer files...")
    # 使用 .save_pretrained() 自动保存所有分词器相关文件，更稳健
    tokenizer.save_pretrained(new_model_path)
    print(f"Tokenizer files saved to '{new_model_path}'.")
    
    print("\n🎉 All Done!")
    print(f"Pruned model has been successfully saved to '{new_model_path}'")


def prune_and_save_model_offline(original_model_path: str, new_model_path: str, layers_to_delete: set[int], layers_path_str: str):
    # ... 函数体无变化 ...
    print("\n" + "="*50)
    print("Starting Offline Model Pruning and Saving")
    print("="*50)
    state_dict = load_model_weights(original_model_path)
    print("Processing weights: removing specified layers and re-indexing...")
    escaped_layers_path = layers_path_str.replace('.', r'\.')
    layer_pattern = re.compile(rf"^{escaped_layers_path}\.(\d+)\.")
    new_state_dict = OrderedDict()
    layers_deleted_count = set()
    for key, value in state_dict.items():
        match = layer_pattern.match(key)
        if not match:
            new_state_dict[key] = value
            continue
        layer_idx = int(match.group(1))
        if layer_idx in layers_to_delete:
            layers_deleted_count.add(layer_idx)
            continue
        num_layers_deleted_before = sum(1 for deleted_idx in layers_to_delete if deleted_idx < layer_idx)
        new_layer_idx = layer_idx - num_layers_deleted_before
        old_prefix = f"{layers_path_str}.{layer_idx}."
        new_prefix = f"{layers_path_str}.{new_layer_idx}."
        new_key = key.replace(old_prefix, new_prefix, 1)
        new_state_dict[new_key] = value
    print(f"Processing complete. Removed {len(layers_deleted_count)} layer(s): {sorted(list(layers_deleted_count))}.")
    print("Updating configuration file...")
    os.makedirs(new_model_path, exist_ok=True)
    config_path = os.path.join(original_model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    if "num_hidden_layers" in config:
        original_num_layers = config["num_hidden_layers"]
        new_num_layers = original_num_layers - len(layers_to_delete)
        config["num_hidden_layers"] = new_num_layers
        print(f"  - Number of layers changed from {original_num_layers} to {new_num_layers}.")
    else:
        print("  - Warning: 'num_hidden_layers' not found in config.json. Cannot update layer count.")
    new_config_path = os.path.join(new_model_path, "config.json")
    with open(new_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    print(f"New config saved to '{new_config_path}'.")
    print("Saving pruned weight file...")
    new_weights_path = os.path.join(new_model_path, "model_state.pdparams")
    paddle.save(new_state_dict, new_weights_path)
    print(f"New weights saved to '{new_weights_path}'.")
    print("Copying tokenizer and other necessary files...")
    files_to_copy = ["tokenizer_config.json", "sentencepiece.bpe.model", "special_tokens_map.json", "generation_config.json", "tokenizer.json", "vocab.txt"]
    for filename in files_to_copy:
        src = os.path.join(original_model_path, filename)
        dst = os.path.join(new_model_path, filename)
        if os.path.exists(src):
            try:
                copyfile(src, dst)
                print(f"  - Copied {filename}")
            except IOError as e:
                print(f"  - Warning: Could not copy {filename}. Error: {e}")
    print("\n🎉 All Done!")
    print(f"Pruned model has been successfully saved to '{new_model_path}'")


# =====================================================================================
# 4. 主执行逻辑
# =====================================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Calculate layer importance, prune, and save a new PaddlePaddle model."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or HuggingFace name of the source PaddlePaddle model.")
    parser.add_argument("--output_model_path", type=str, required=True, help="Path to save the new, pruned model directory.")
    parser.add_argument("--layers_path", type=str, required=True, help="Dot-separated path to the layers list (e.g., 'llama.layers').")
    parser.add_argument("--n_prune_layers", type=int, required=True, help="The number of layers to identify and prune.")
    parser.add_argument("--dataset_name", type=str, default="emozilla/pg19", help="Name of the Hugging Face dataset for calibration. Default: 'emozilla/pg19'.")
    parser.add_argument("--dataset_split", type=str, default="validation", help="The split of the dataset to use. Default: 'validation'.")
    args = parser.parse_args()

    # --- PART 1: 计算层重要性 ---
    print("="*50)
    print("PART 1: Calculating Layer Importance")
    print("="*50)
    print(f"Loading '{args.dataset_split}' split from '{args.dataset_name}' dataset for calibration...")
    try:
        data = load_dataset(args.dataset_name, split=args.dataset_split)
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        print("Please ensure the dataset name and split are correct and you have internet access for Hugging Face datasets.")
        return
    
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    
    short_model = ShortGPT(model_name=args.model_name_or_path, layers_path=args.layers_path)
    
    print("=== 调试模型路径信息 ===")
    print(f"Model type: {type(short_model)}")
    print(f"Tokenizer type: {type(short_model.tokenizer)}")

    # 检查config中的路径
    if hasattr(short_model, 'config'):
        if hasattr(short_model.config, 'name_or_path'):
            print(f"Config path: {short_model.config.name_or_path}")
        if hasattr(short_model.config, '_name_or_path'):
            print(f"Config _name_or_path: {short_model.config._name_or_path}")

    # 检查tokenizer中的路径
    if hasattr(short_model.tokenizer, 'name_or_path'):
        print(f"Tokenizer path: {short_model.tokenizer.name_or_path}")
        
        
    for batch in tqdm(dataloader, desc="Evaluating Layer Importance"):
        if 'text' not in batch:
            raise ValueError("Dataset must contain a 'text' column.")
        prompts = batch['text']
        short_model.eval_importance(prompts=prompts, stride=256)

    # 根据重要性分数排序层索引（越小越不重要）
    prune_order = [i for i, _ in sorted(enumerate(short_model.importances), key=lambda x: x[1])]
    
    print("\n--- Importance Calculation Complete ---")
    print(f"Calculated importances: {[f'{v:.2f}' for v in short_model.importances]}")
    print(f"Pruning order (least to most important): {prune_order}")
    
    # 确定要删除的层
    layers_to_delete = set(prune_order[:args.n_prune_layers])
    print(f"Will delete the {args.n_prune_layers} least important layers: {sorted(list(layers_to_delete))}")

    # --- PART 2: 直接在内存中执行剪枝与保存 ---
    prune_and_save_model_in_memory(
        model=short_model.model,
        tokenizer=short_model.tokenizer,
        new_model_path=args.output_model_path,
        layers_to_delete=layers_to_delete,
        layers_path_str=args.layers_path
    )

if __name__ == "__main__":
    main()