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
import argparse
import copy
import os

import paddle

from paddlenlp.peft import LoRAConfig, LoRAModel

try:
    from paddle.nn.quant import weight_dequantize, weight_quantize
except:
    weight_dequantize = None
    weight_quantize = None
try:
    from paddlenlp.quantization.qlora import qlora_weight_quantize_dequantize
except:
    qlora_weight_quantize_dequantize = None

from paddlenlp.quantization.quantization_config import QuantizationConfig
from paddlenlp.transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from paddlenlp.transformers.utils import device_guard
from paddlenlp.utils.env import CONFIG_NAME


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, help="The directory of pretrained model.")
    parser.add_argument(
        "--lora_path", default=None, required=True, help="The directory of LoRA parameters. Default to None"
    )
    parser.add_argument(
        "--merge_lora_model_path",
        default=None,
        required=True,
        help="The directory of merged parameters. Default to None",
    )
    parser.add_argument("--device", type=str, default="gpu", help="Device")
    parser.add_argument(
        "--low_gpu_mem", type=bool, default=False, help="Whether to use low gpu memory. Default to False"
    )
    return parser.parse_args()


def weight_process(name, quant_config, lora_config, state_dict):
    weight = state_dict.pop(name + ".weight").cuda()
    if quant_config.weight_quantize_algo is None:
        pass
    elif quant_config.weight_quantize_algo in ["nf4", "fp4"]:
        weight = qlora_weight_quantize_dequantize(
            weight,
            quant_algo=quant_config.weight_quantize_algo,
            double_quant=quant_config.weight_double_quant,
            block_size=quant_config.weight_blocksize,
            double_quant_block_size=quant_config.weight_double_quant_block_size,
        )
    elif quant_config.weight_quantize_algo in ["weight_only_int8"]:
        out, scale = weight_quantize(weight, algo=quant_config.weight_quantize_algo)
        weight = weight_dequantize(out, scale)
    else:
        raise ValueError(f"quant_config.weight_quantize_algo {quant_config.weight_quantize_algo} is not supported.")
    lora_A = state_dict.pop(name + ".lora_A").cuda()
    lora_B = state_dict.pop(name + ".lora_B").cuda()
    scaling = lora_config.lora_alpha / lora_config.r
    state_dict[name + ".weight"] = (weight + lora_A @ lora_B * scaling).cpu()


def merge():
    args = parse_arguments()
    paddle.set_device(args.device)

    lora_config = LoRAConfig.from_pretrained(args.lora_path)
    if lora_config.base_model_name_or_path is None:
        if args.model_name_or_path is not None:
            raise ValueError("We can not find a valid model_name_or_path.")
        else:
            lora_config.base_model_name_or_path = args.model_name_or_path

    if os.path.isfile(os.path.join(args.lora_path, CONFIG_NAME)):
        config = AutoConfig.from_pretrained(args.lora_path)
    elif args.model_name_or_path is not None:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            f"We can not find config.json in lora_path: {args.lora_path} or find a valid model_name_or_path."
        )
    config.dtype = lora_config.dtype
    if (
        lora_config.dtype == "bfloat16" or config.quantization_config.weight_quantize_algo in ["nf4", "fp4"]
    ) and args.device == "cpu":
        raise ValueError("We can not apply bfloat16 or nf4/fp4 lora merge on cpu.")

    if args.low_gpu_mem and args.device == "gpu":
        quant_config = copy.deepcopy(config.quantization_config)
        config.quantization_config = QuantizationConfig()
        lora_config.merge_weights = False
        with device_guard():
            model = AutoModelForCausalLM.from_pretrained(
                lora_config.base_model_name_or_path,
                config=config,
                low_cpu_mem_usage=True,
            )
            model = LoRAModel.from_pretrained(model=model, lora_path=args.lora_path, lora_config=lora_config)
        model.eval()
        model_state_dict = model.model.state_dict()
        lora_name_list = []
        for key in model_state_dict.keys():
            if "lora_A" in key:
                lora_name_list.append(key[:-7])
        for name in lora_name_list:
            weight_process(name, quant_config, lora_config, model_state_dict)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            lora_config.base_model_name_or_path,
            config=config,
            low_cpu_mem_usage=args.low_gpu_mem,
        )
        lora_config.merge_weights = True
        model = LoRAModel.from_pretrained(model=model, lora_path=args.lora_path, lora_config=lora_config)
        model.eval()
        model_state_dict = model.model.state_dict()
        for key in list(model_state_dict):
            if "lora" in key:
                del model_state_dict[key]
            if "quant" in key:
                del model_state_dict[key]
        model.model.config.quantization_config = QuantizationConfig()
    model.model.save_pretrained(args.merge_lora_model_path, state_dict=model_state_dict)

    tokenizer = AutoTokenizer.from_pretrained(lora_config.base_model_name_or_path)
    tokenizer.save_pretrained(args.merge_lora_model_path)


if __name__ == "__main__":
    merge()
