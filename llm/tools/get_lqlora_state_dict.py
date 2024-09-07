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
from utils.utils import get_lora_target_modules

from paddlenlp.peft import LoRAConfig, LoRAModel
from paddlenlp.peft.lora.lqlora_utils import transform_lora_layers
from paddlenlp.transformers import AutoModelForCausalLM


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, required=True, type=str, help="The directory of model.")
    parser.add_argument(
        "--lqlora_quantize_cfg", default=None, type=str, required=True, help="The directory of lqlora quantize config"
    )
    parser.add_argument("--output_path", default=None, type=str, required=True, help="The directory of saved model ")
    return parser.parse_args()


def get_lqlora_state_dict():
    args = parse_arguments()
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model = model.to(dtype=paddle.float16)

    target_modules = get_lora_target_modules(model)
    lora_config = LoRAConfig(
        target_modules=target_modules,
        r=8,
        lora_alpha=16,
        merge_weights=False,
        tensor_parallel_degree=1,
        dtype=paddle.float16,
        base_model_name_or_path=args.model_name_or_path,
    )
    model = LoRAModel(model, lora_config)
    lqlora_quantize_cfg = paddle.load(args.lqlora_quantize_cfg)
    transform_lora_layers(model, lqlora_quantize_cfg)

    state_dict = model.state_dict()
    paddle.save(state_dict, args.output_path)


if __name__ == "__main__":
    get_lqlora_state_dict()
