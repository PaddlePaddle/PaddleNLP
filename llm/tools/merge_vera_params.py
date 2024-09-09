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

from paddlenlp.peft import VeRAConfig, VeRAModel
from paddlenlp.transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from paddlenlp.utils.env import CONFIG_NAME


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, help="The directory of pretrained model.")
    parser.add_argument("--vera_path", default="", help="The directory of VeRA parameters. Default to None")
    parser.add_argument(
        "--merge_vera_model_path",
        default="",
        help="The directory of merged parameters. Default to None",
    )
    parser.add_argument("--device", type=str, default="gpu", help="Device")
    parser.add_argument(
        "--low_gpu_mem", type=bool, default=True, help="Whether to use low gpu memory. Default to False"
    )
    return parser.parse_args()


def weight_process(name, vera_config, state_dict):
    weight = state_dict.pop(name + ".weight").cuda()
    vera_A = state_dict.pop(name + ".vera_A").cuda()
    vera_B = state_dict.pop(name + ".vera_B").cuda()
    vera_b = state_dict.pop(name + ".vera_b").cuda()
    vera_d = state_dict.pop(name + ".vera_d").cuda()
    diag_b = paddle.diag(vera_b)
    diag_d = paddle.diag(vera_d)

    scaling = vera_config.vera_alpha / vera_config.r
    state_dict[name + ".weight"] = (weight + vera_A @ diag_d @ vera_B @ diag_b * scaling).cpu()


def merge():
    args = parse_arguments()
    paddle.set_device(args.device)

    vera_config = VeRAConfig.from_pretrained(args.vera_path)
    if vera_config.base_model_name_or_path is None:
        if args.model_name_or_path is not None:
            raise ValueError("We can not find a valid model_name_or_path.")
        else:
            vera_config.base_model_name_or_path = args.model_name_or_path

    if os.path.isfile(os.path.join(args.vera_path, CONFIG_NAME)):
        config = AutoConfig.from_pretrained(args.vera_path)
    elif args.model_name_or_path is not None:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            f"We can not find config.json in vera_path: {args.vera_path} or find a valid model_name_or_path."
        )
    config.dtype = vera_config.dtype
    if (
        vera_config.dtype == "bfloat16" or config.quantization_config.weight_quantize_algo in ["nf4", "fp4"]
    ) and args.device == "cpu":
        raise ValueError("We can not apply bfloat16 or nf4/fp4 vera merge on cpu.")

    # with device_guard() will cause SVD decomposition to fail
    model = AutoModelForCausalLM.from_pretrained(
        vera_config.base_model_name_or_path,
        config=config,
        low_cpu_mem_usage=True,
    )
    model = VeRAModel.from_pretrained(model=model, vera_path=args.vera_path, vera_config=vera_config)

    model.eval()
    model_state_dict = model.model.state_dict()
    vera_name_list = []
    for key in model_state_dict.keys():
        if "vera_A" in key:
            vera_name_list.append(key[:-7])

    for name in vera_name_list:
        weight_process(name, vera_config, model_state_dict)

    model.model.save_pretrained(args.merge_vera_model_path, state_dict=model_state_dict)
    tokenizer = AutoTokenizer.from_pretrained(vera_config.base_model_name_or_path)
    tokenizer.save_pretrained(args.merge_vera_model_path)


if __name__ == "__main__":
    merge()
