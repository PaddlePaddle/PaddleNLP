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
import json
import os

import numpy as np
import paddle

from paddlenlp.generation import GenerationConfig
from paddlenlp.transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from paddlenlp.transformers.model_utils import load_tp_checkpoint
from paddlenlp.trl import llm_utils


def parse_arguments():
    """
    parse_arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None, type=str, required=True, help="The directory of model.")
    parser.add_argument("--output_path", default=None, type=str, help="The directory of splited model")
    parser.add_argument("--model_rank_id", default=None, type=int, help="Input model mp degree.")
    parser.add_argument("--dtype", default="float16", type=str, help="The dtype of model weights.")
    return parser.parse_args()


def split(args):
    """
    Split model weight
    """
    rank, nranks = llm_utils.init_dist_env()

    if args.output_path is None:
        args.output_path = os.path.join(args.model_path, f"{nranks}_ranks")

    paddle.set_default_dtype(args.dtype)

    config = AutoConfig.from_pretrained(args.model_path)
    config.tensor_parallel_degree = nranks
    config.tensor_parallel_rank = rank

    generation_config = GenerationConfig.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.model_rank_id is not None:
        model_path = os.path.join(args.model_path, f"model_state.tp0{args.model_rank_id - 1}.pdparams")
        assert os.path.isfile(model_path), f"{model_path} not exist"
        state_dict = load_tp_checkpoint(args.model_path, model, config)
        model_rank = args.model_rank_id
        save_base_rank = model_rank * nranks
    else:
        state_dict = load_tp_checkpoint(args.model_path, model, config)
        model_rank = 0
        save_base_rank = 0

    weight_file = os.path.join(args.output_path, f"model_state.tp0{rank + save_base_rank}.pdparams")
    paddle.save(state_dict, weight_file)

    # process weight scales
    possible_weight_scales_path = os.path.join(args.model_path, f"weight_scales_{model_rank}.json")
    if os.path.exists(possible_weight_scales_path) and rank == 0:
        with open(possible_weight_scales_path, "r") as f:
            weight_scales_dict = json.load(f)

        processed_weight_scales = [{} for i in range(nranks)]
        for k, v in weight_scales_dict.items():
            if "self_attn.q_proj" in k:
                splited_value = np.split(np.array(v), nranks, axis=-1)
                for tp_rank in range(nranks):
                    processed_weight_scales[tp_rank][k] = splited_value[tp_rank].tolist()
            elif "self_attn.k_proj" in k:
                splited_value = np.split(np.array(v), nranks, axis=-1)
                for tp_rank in range(nranks):
                    processed_weight_scales[tp_rank][k] = splited_value[tp_rank].tolist()
            elif "self_attn.v_proj" in k:
                splited_value = np.split(np.array(v), nranks, axis=-1)
                for tp_rank in range(nranks):
                    processed_weight_scales[tp_rank][k] = splited_value[tp_rank].tolist()
            elif "self_attn.o_proj" in k:
                for tp_rank in range(nranks):
                    processed_weight_scales[tp_rank][k] = v
            elif "mlp.gate_proj" in k:
                splited_value = np.split(np.array(v), nranks, axis=-1)
                for tp_rank in range(nranks):
                    processed_weight_scales[tp_rank][k] = splited_value[tp_rank].tolist()
            elif "mlp.up_proj" in k:
                splited_value = np.split(np.array(v), nranks, axis=-1)
                for tp_rank in range(nranks):
                    processed_weight_scales[tp_rank][k] = splited_value[tp_rank].tolist()
            elif "mlp.down_proj" in k:
                for tp_rank in range(nranks):
                    processed_weight_scales[tp_rank][k] = v
            else:
                raise ValueError(f"key {k} is not supported!")

        for tp_rank in range(nranks):
            save_path = os.path.join(args.output_path, f"weight_scales_{tp_rank + save_base_rank}.json")
            with open(save_path, "w") as f:
                print("weight scale save_path:", save_path)
                json.dump(processed_weight_scales[tp_rank], f)

    # process cachekv scales
    possible_cache_path = os.path.join(args.model_path, f"cachekv_scales_{model_rank}.json")
    if os.path.exists(possible_cache_path) and rank == 0:
        with open(possible_cache_path, "r") as f:
            cache_dict = json.load(f)

        processed_cachekv_scales = [{} for i in range(nranks)]
        for k, v in cache_dict.items():
            v = np.array(v).flatten()
            splited_value = np.split(np.array(v), nranks, axis=-1)
            for tp_rank in range(nranks):
                processed_cachekv_scales[tp_rank][k] = splited_value[tp_rank].tolist()
        for tp_rank in range(nranks):
            save_path = os.path.join(args.output_path, f"cachekv_scales_{tp_rank + save_base_rank}.json")
            print("cachekv scale save_path:", save_path)
            with open(save_path, "w") as f:
                json.dump(processed_cachekv_scales[tp_rank], f)

    # process act scales
    possible_act_scales_path = os.path.join(args.model_path, f"act_scales_{model_rank}.json")
    if os.path.exists(possible_act_scales_path) and rank == 0:
        with open(possible_act_scales_path, "r") as f:
            act_scale = json.load(f)
        for tp_rank in range(nranks):
            save_path = os.path.join(args.output_path, f"act_scales_{tp_rank + save_base_rank}.json")
            with open(save_path, "w") as outf:
                print("act scale save_path:", save_path)
                json.dump(act_scale, outf)

    if rank == 0:
        tokenizer.save_pretrained(args.output_path)
        config.save_pretrained(args.output_path)
        generation_config.save_pretrained(args.output_path)


if __name__ == "__main__":
    """
    Script to split model weight.
    """
    args = parse_arguments()
    split(args)
