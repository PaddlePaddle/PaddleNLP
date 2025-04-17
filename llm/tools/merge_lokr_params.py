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

from paddlenlp.peft import LoKrConfig, LoKrModel
from paddlenlp.transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from paddlenlp.utils.env import CONFIG_NAME


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, help="The directory of pretrained model.")
    parser.add_argument("--lokr_path", default="", help="The directory of lokr parameters. Default to None")
    parser.add_argument(
        "--merge_lokr_model_path",
        default="",
        help="The directory of merged parameters. Default to None",
    )
    parser.add_argument("--device", type=str, default="gpu", help="Device")
    parser.add_argument(
        "--low_gpu_mem", type=bool, default=True, help="Whether to use low gpu memory. Default to False"
    )
    return parser.parse_args()


def weight_process(name, lokr_config, state_dict):
    weight = state_dict.pop(name + ".weight")
    use_w1 = True if ((name + ".lokr_w1") in state_dict) else False
    use_w2 = True if ((name + ".lokr_w2") in state_dict) else False
    if use_w1:
        lokr_w1 = state_dict.pop(name + ".lokr_w1")
    else:
        lokr_w1_a = state_dict.pop(name + ".lokr_w1_a")
        lokr_w1_b = state_dict.pop(name + ".lokr_w1_b")
    if use_w2:
        lokr_w2 = state_dict.pop(name + ".lokr_w2")
    else:
        lokr_w2_a = state_dict.pop(name + ".lokr_w2_a")
        lokr_w2_b = state_dict.pop(name + ".lokr_w2_b")

    scaling = lokr_config.lokr_alpha / lokr_config.lokr_dim

    adapter_weight = (
        scaling
        * paddle.kron(lokr_w1 if use_w1 else lokr_w1_a @ lokr_w1_b, lokr_w2 if use_w2 else lokr_w2_a @ lokr_w2_b).T
    )
    state_dict[name + ".weight"] = weight + adapter_weight


def merge():
    args = parse_arguments()
    paddle.set_device(args.device)

    lokr_config = LoKrConfig.from_pretrained(args.lokr_path)
    if lokr_config.base_model_name_or_path is None:
        if args.model_name_or_path is not None:
            raise ValueError("We can not find a valid model_name_or_path.")
        else:
            lokr_config.base_model_name_or_path = args.model_name_or_path

    if os.path.isfile(os.path.join(args.lokr_path, CONFIG_NAME)):
        config = AutoConfig.from_pretrained(args.lokr_path)
    elif args.model_name_or_path is not None:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            f"We can not find config.json in lokr_path: {args.lokr_path} or find a valid model_name_or_path."
        )
    config.dtype = lokr_config.dtype
    if (
        lokr_config.dtype == "bfloat16" or config.quantization_config.weight_quantize_algo in ["nf4", "fp4"]
    ) and args.device == "cpu":
        raise ValueError("We can not apply bfloat16 or nf4/fp4 lokr merge on cpu.")

    # with device_guard() will cause SVD decomposition to fail
    model = AutoModelForCausalLM.from_pretrained(
        lokr_config.base_model_name_or_path,
        config=config,
        low_cpu_mem_usage=True,
    )
    model = LoKrModel.from_pretrained(model=model, lokr_path=args.lokr_path, lokr_config=lokr_config)

    model.eval()
    model_state_dict = model.model.state_dict()
    lokr_name_list = []

    for key in model_state_dict.keys():
        if "lokr" in key:
            lokr_name_list.append(key.split(".lokr")[0])

    lokr_name_list = list(set(lokr_name_list))
    for name in lokr_name_list:
        weight_process(name, lokr_config, model_state_dict)

    model.model.save_pretrained(args.merge_lokr_model_path, state_dict=model_state_dict)
    tokenizer = AutoTokenizer.from_pretrained(lokr_config.base_model_name_or_path)
    tokenizer.save_pretrained(args.merge_lokr_model_path)


if __name__ == "__main__":
    merge()
