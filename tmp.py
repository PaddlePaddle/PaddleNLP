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


import os
import re

import wandb

os.environ["WANDB_MODE"] = "offline"

log_project = "Safe-RLHF-RM"
log_run_name = "paddle-1102"
log_dir = "./paddle-wandb"
config = {}
self_wandb = wandb.init(
    project=log_project,
    name=log_run_name,
    dir=log_dir,
    config=config,
)

log_path = "/root/paddlejob/workspace/guosheng/share/paddlenlp-rlhf/PaddleNLP/examples/RLHF/log/workerlog.0"
with open(log_path, "r") as f:
    global_step = None
    for line in f.readlines():
        if "eval_accuracy:" in line:
            # print(line.strip())
            pattern = re.compile(
                r"eval_accuracy: (\d+\.\d+), eval_rewards_mean: -?(\d+\.\d+), eval_rewards_std: (\d+\.\d+),"
            )
            rs = re.search(pattern, line)
            if rs:
                eval_accuracy = rs.group(1)
                eval_rewards_mean = rs.group(2)
                eval_rewards_std = rs.group(3)
                print(eval_accuracy, eval_rewards_mean, eval_rewards_std)
                metrics = {
                    "eval/accuracy": float(eval_accuracy),
                    "eval/rewards_mean": float(eval_rewards_mean),
                    "eval/rewards_std": float(eval_rewards_std),
                }
                # print(metrics)
                self_wandb.log(metrics, step=int(global_step))
        elif " loss: " in line:
            # print(line.strip())
            pattern = re.compile(
                # r"loss: (\d+\.\d+), learning_rate: (\d+\.\d+), global_step: (\d+),.+accuracy: (\d+\.\d+),"
                r"loss: (\d+\.\d+), learning_rate: (\d+)(\.\d+)?(e-\d+)?, global_step: (\d+),.+accuracy: (\d+\.\d+),"
            )
            rs = re.search(pattern, line)
            if rs:
                loss = rs.group(1)
                # print(rs.groups())
                learning_rate = (
                    rs.group(2) + (rs.group(3) if rs.group(3) else "") + (rs.group(4) if rs.group(4) else "")
                )
                global_step = rs.group(5)
                accuracy = rs.group(6)
                # print(loss, accuracy, learning_rate, global_step)
                metrics = {
                    "train/loss": float(loss),
                    "train/accuracy": float(accuracy),
                    "train/lr": float(learning_rate),
                }
                # print(metrics)
                self_wandb.log(metrics, step=int(global_step))
                # exit(0)
self_wandb.finish()

exit(0)

import os

import numpy as np
import paddle

# import torch
import transformers

from paddlenlp.transformers import (  # AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModelForScore,
)
from paddlenlp.utils.log import logger

# from datasets import load_dataset


# for alignment ####
"""
ids_data = np.load(
    "/root/paddlejob/workspace/guosheng/share/safe-rlhf/ids.npy")
mask_data = np.load(
    "/root/paddlejob/workspace/guosheng/share/safe-rlhf/mask.npy")
print(ids_data, ids_data.shape, ids_data.dtype)
print(mask_data, mask_data.shape, mask_data.dtype)
input_ids = paddle.to_tensor(ids_data)
attention_mask = paddle.to_tensor(mask_data)
# pd_model = LlamaForCausalLM.from_pretrained(
#     "/root/paddlejob/workspace/guosheng/alpaca-7b-reproduced-saved/",
#     # "/root/paddlejob/workspace/guosheng/alpaca-7b-reproduced",
#     dtype="float32")
pd_model = LlamaModelForScore.from_pretrained(
    # "/root/paddlejob/workspace/guosheng/alpaca-7b-reproduced",
    "/root/paddlejob/workspace/guosheng/alpaca-7b-reproduced-saved/",
    dtype="float32")
pd_model = pd_model.get_decoder()
out = pd_model(input_ids, attention_mask=attention_mask)
hidden_state = out[0]
print(hidden_state.numpy(), hidden_state.shape, hidden_state.dtype)
del pd_model, out, input_ids, attention_mask
torch.backends.cuda.matmul.allow_tf32 = False
input_ids = torch.LongTensor(ids_data).cuda()
attention_mask = torch.Tensor(mask_data).cuda()
th_model = transformers.LlamaForCausalLM.from_pretrained(
    "PKU-Alignment/alpaca-7b-reproduced", torch_dtype=torch.float32).cuda()
th_model = th_model.get_decoder()
out = th_model(input_ids, attention_mask=attention_mask)
hidden_state = out[0]
print(hidden_state.detach().cpu().numpy(), hidden_state.shape, hidden_state.dtype)
exit(0)
"""

# for tokenizer ####
# load_dataset("PKU-Alignment/PKU-SafeRLHF", split="30k_train")

# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     "/root/paddlejob/workspace/guosheng/alpaca-7b-reproduced-hf",
#     use_fast=False)
# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     "PKU-Alignment/alpaca-7b-reproduced", use_fast=False)
# tokenizer.save_pretrained(
#     "/root/paddlejob/workspace/guosheng/alpaca-7b-reproduced-hf")
# tokenizer = AutoTokenizer.from_pretrained("/root/paddlejob/workspace/guosheng/alpaca-7b-reproduced-saved/")
# tokenizer = AutoTokenizer.from_pretrained("facebook/llama-7b")
# print("="*20, tokenizer.pad_token)
# tokenizer.save_pretrained("/root/paddlejob/workspace/guosheng/alpaca-7b-reproduced-pd")

# exit(0)
# model = LlamaForCausalLM.from_pretrained("facebook/llama-7b")
# state_dict = model.state_dict()
# for k, v in state_dict.items():
#     print(k, v.dtype)
# exit(0)


# for merge LlamaForCausalLM weight and score_head.weight ####
input_ids = paddle.to_tensor(np.load("/root/paddlejob/workspace/guosheng/share/safe-rlhf/ids.npy"))
attention_mask = paddle.to_tensor(np.load("/root/paddlejob/workspace/guosheng/share/safe-rlhf/mask.npy"))
score_head_weight = paddle.to_tensor(
    np.load(
        # "/root/paddlejob/workspace/guosheng/share/safe-rlhf/score_head_weight.npy"，
        "/root/paddlejob/workspace/guosheng/share/safe-rlhf/bf16_score_head_weight.npy"
    ).transpose(1, 0)
)
pd_model = LlamaModelForScore.from_pretrained(
    "/root/paddlejob/workspace/guosheng/alpaca-7b-reproduced", dtype="float32"
)
pd_model.score_head.weight.set_value(score_head_weight.cast(pd_model.score_head.weight.dtype))
pd_model.save_pretrained("/root/paddlejob/workspace/guosheng/alpaca-7b-reproduced-bf16-head-saved")
with paddle.no_grad():
    output = pd_model(input_ids, attention_mask=attention_mask, return_dict=True)
# th_model = transformers.LlamaForCausalLM.from_pretrained(
#     "PKU-Alignment/alpaca-7b-reproduced")
print(pd_model.get_decoder().embed_tokens.weight)
scores = output.scores  # size = (2 * B, L, 1)
end_scores = output.end_scores  # size = (2 * B, 1)
print(scores)
print(end_scores)
# print(th_model.get_decoder().embed_tokens.weight, th_model.get_decoder().embed_tokens.weight.dtype)
# state_dict = paddle.load(
#     "/root/paddlejob/workspace/guosheng/alpaca-7b-reproduced/model_state.pdparams")
# for k, v in state_dict.items():
#     print(k, v.dtype)
exit(0)


def convert(cls, weight_file, config, cache_dir: str) -> None:
    """the entry of converting config and converting model file

    Args:
        input_dir (str | None): the input dir which contains `pytorch_model.bin` and `config.json` file
        config (PretrainedConfig): the PretrainedConfig instance of model
    """
    # FIXME(wj-Mcat): add compatibility with downstream models

    name_mappings = cls._get_name_mappings(config)

    state_dict = transformers.LlamaForCausalLM.from_pretrained(weight_file).state_dict()
    for k, v in state_dict.items():
        state_dict[k] = v.numpy()

    # 3. convert state_dict
    all_layer_names = set(state_dict.keys())
    for name_mapping in name_mappings:
        print("=" * 20, name_mapping.source_name, name_mapping.target_name)
        if name_mapping.source_name not in state_dict:
            logger.warning(f"key<{name_mapping.source_name}> not in the pytorch weight file.")
            continue

        state_dict[name_mapping.target_name] = name_mapping.run(state_dict, name_mapping.source_name)
        if name_mapping.source_name in all_layer_names:
            all_layer_names.remove(name_mapping.source_name)

    if all_layer_names:
        logger.warning(f"there are {len(all_layer_names)} tensors not initialized:")
        for layer_name in all_layer_names:
            logger.warning(f"--- {layer_name}")

    model_weight_file = os.path.join(cache_dir, "model.pdparams")
    paddle.save(state_dict, model_weight_file)
    return state_dict


config = LlamaConfig.from_pretrained("./hf_config.json")
# config.save_pretrained("./")
convert(LlamaForCausalLM, weight_file="PKU-Alignment/alpaca-7b-reproduced", config=config, cache_dir="./")

# state_dict = paddle.load("model.pdparams")
# for k, v in state_dict.items():
#     print(k, v.dtype)
exit(0)
