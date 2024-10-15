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
import sys

sys.path.append(os.path.dirname(os.getcwd()))

import paddle

from paddlenlp.peft import LoRAConfig, LoRAModel
from paddlenlp.peft.lora.lqlora_utils import transform_lora_layers
from paddlenlp.transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from paddlenlp.utils.llm_utils import get_lora_target_modules


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True, type=str, help="The directory of model.")
    parser.add_argument("--dtype", default="float16", type=str, help="The data type of tensor")
    parser.add_argument("--lora_rank", default="8", type=int, help="Lora attention dimension")
    parser.add_argument("--tensor_parallel_degree", default="1", type=int, help="1 for not use tensor parallel")
    parser.add_argument(
        "--lqlora_quantize_cfg", type=str, required=True, help="The directory of lqlora quantize config"
    )
    parser.add_argument("--output_path", type=str, required=True, help="The directory of saved model ")
    return parser.parse_args()


def get_lqlora_state_dict():
    args = parse_arguments()
    base_model_dir = os.path.join(args.output_path, "backbone")
    lora_model_dir = os.path.join(args.output_path, "adapter")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    model_config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        dtype=args.dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=model_config)

    target_modules = get_lora_target_modules(model)
    lora_config = LoRAConfig(
        target_modules=target_modules,
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        merge_weights=False,
        tensor_parallel_degree=args.tensor_parallel_degree,
        dtype=args.dtype,
        base_model_name_or_path=args.model_name_or_path,
    )
    model = LoRAModel(model, lora_config)

    lqlora_quantize_cfg = paddle.load(args.lqlora_quantize_cfg)
    transform_lora_layers(model, lqlora_quantize_cfg)

    model.model.save_pretrained(base_model_dir)
    tokenizer.save_pretrained(base_model_dir)
    model.save_pretrained(lora_model_dir)


if __name__ == "__main__":
    get_lqlora_state_dict()
