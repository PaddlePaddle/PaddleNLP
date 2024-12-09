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
import copy
import math
import os

import numpy as np
import paddle
from paddle.nn.quant import weight_dequantize

from paddlenlp.peft import LoRAConfig, LoRAModel

try:
    from paddlenlp.quantization.qlora import qlora_weight_quantize_dequantize
    from paddlenlp.quantization.quantization_config import QuantizationConfig
    from paddlenlp.quantization.quantization_linear import QuantizationLinear
except:
    pass

from paddlenlp.trainer.argparser import strtobool
from paddlenlp.transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from paddlenlp.transformers.utils import device_guard
from paddlenlp.utils.env import CONFIG_NAME
from paddlenlp.utils.log import logger


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, required=True, type=str, help="The directory of model.")
    parser.add_argument(
        "--lora_path", default=None, type=str, required=True, help="The directory of LoRA parameters. Default to None"
    )
    parser.add_argument("--output_path", default=None, type=str, required=True, help="The directory of saved model ")
    parser.add_argument("--safe_serialization", default="False", type=strtobool, help="Whether save as safetensor.")
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "npu", "cpu"],
        help="Device for selecting for merging lora weights, currently only supports gpu/npu/cpu.",
    )
    return parser.parse_args()


def weight_process(name, quant_config, lora_config, state_dict, device):
    target_device = device if device == "cpu" else device + ":0"

    if (name + ".weight") not in state_dict.keys():
        return

    if quant_config.weight_quantize_algo is None:
        return
    elif quant_config.weight_quantize_algo in ["nf4", "fp4"]:
        weight = state_dict.pop(name + ".weight").to(target_device)
        state_dict[name + ".weight"] = qlora_weight_quantize_dequantize(
            weight,
            quant_algo=quant_config.weight_quantize_algo,
            double_quant=quant_config.weight_double_quant,
            block_size=quant_config.weight_blocksize,
            double_quant_block_size=quant_config.weight_double_quant_block_size,
        ).cpu()
    elif quant_config.weight_quantize_algo in ["weight_only_int8"]:
        quant_weight = state_dict.pop(name + ".quant_weight").to(target_device)
        quant_scale = state_dict.pop(name + ".quant_scale").to(target_device)
        state_dict[name + ".weight"] = weight_dequantize(quant_weight, quant_scale, out_dtype=lora_config.dtype).cpu()
    else:
        raise ValueError(f"quant_config.weight_quantize_algo {quant_config.weight_quantize_algo} is not supported.")


def lora_process(name, lora_config, state_dict, device, lora_state_dict=None):
    target_device = device if device == "cpu" else device + ":0"

    if (name + ".weight") not in state_dict.keys():
        return

    weight = state_dict.pop(name + ".weight")
    lora_use_mixer = lora_config.lora_use_mixer
    if lora_state_dict is None:
        lora_A = state_dict.pop(name + ".lora_A")
        lora_B = state_dict.pop(name + ".lora_B")
        if lora_use_mixer:
            lora_AB = state_dict.pop(name + ".lora_AB")
    else:
        lora_A = lora_state_dict.pop(name + ".lora_A")
        lora_B = lora_state_dict.pop(name + ".lora_B")
        if lora_use_mixer:
            lora_AB = lora_state_dict.pop(name + ".lora_AB")
    if device != "cpu":
        weight = weight.to(target_device)
        lora_A = lora_A.to(target_device)
        lora_B = lora_B.to(target_device)
        if lora_use_mixer:
            lora_AB = lora_AB.to(target_device)
    if not lora_config.rslora:
        scaling = lora_config.lora_alpha / lora_config.r
    else:
        scaling = lora_config.lora_alpha / math.sqrt(lora_config.r)

    if device == "cpu" and weight.dtype.name == "BF16":
        weight = weight.astype("float32")
        lora_A = lora_A.astype("float32")
        lora_B = lora_B.astype("float32")
        if lora_use_mixer:
            lora_AB = lora_AB.astype(lora_config.dtype)
            out = (weight + lora_A @ lora_AB @ lora_B * scaling).astype(lora_config.dtype)
        else:
            out = (weight + lora_A @ lora_B * scaling).astype(lora_config.dtype)
    else:
        if lora_use_mixer:
            out = (weight + lora_A @ lora_AB @ lora_B * scaling).cpu()
        else:
            out = (weight + lora_A @ lora_B * scaling).cpu()

    state_dict[name + ".weight"] = out


def merge_old_lora(lora_config, args):
    lora_config.merge_weights = True
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=lora_config.dtype,
    )
    model = LoRAModel.from_pretrained(model, args.lora_path)
    try:
        model.merge()
        model.eval()
    except:
        model.eval()
    model_state_dict = model.model.state_dict()
    for key in list(model_state_dict):
        if "lora" in key:
            del model_state_dict[key]
    return model, model_state_dict


def read_file(file_name):
    if file_name.endswith("safetensors"):
        try:
            from paddlenlp.utils.safetensors import fast_load_file as load_file
        except:
            from safetensors.numpy import load_file

        read_tensors = load_file(file_name)
        for key in list(read_tensors.keys()):
            if isinstance(read_tensors[key], np.ndarray):
                with device_guard("cpu"):
                    read_tensors[key] = paddle.Tensor(read_tensors.pop(key), zero_copy=True)
    else:
        with device_guard("cpu"):
            read_tensors = paddle.load(file_name)
    return read_tensors


def save_file(output_path, file_name, tensors, safe_serialization=True):
    if safe_serialization:
        from safetensors.numpy import save_file as _save_file

        if file_name == "model_state.pdparams":
            file_name = "model.safetensors"

        for key in list(tensors.keys()):
            if isinstance(tensors[key], paddle.Tensor):
                tensors[key] = tensors.pop(key).cpu().numpy()
        _save_file(tensors, os.path.join(output_path, file_name), metadata={"format": "np"})
    else:
        paddle.save(tensors, os.path.join(output_path, file_name))


def merge():
    args = parse_arguments()
    paddle.set_device(args.device)

    lora_config = LoRAConfig.from_pretrained(args.lora_path)
    if os.path.isfile(os.path.join(args.lora_path, CONFIG_NAME)):
        config = AutoConfig.from_pretrained(args.lora_path)
    elif args.model_name_or_path is not None:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            f"We can not find config.json in lora_path: {args.lora_path} or find a valid model_name_or_path."
        )
    config.dtype = lora_config.dtype
    quant_config = copy.deepcopy(config.quantization_config)
    lora_config.merge_weights = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.save_pretrained(args.output_path)

    if lora_config.enable_lora_list is not None:
        model, model_state_dict = merge_old_lora(lora_config, args)
    else:
        if quant_config.weight_quantize_algo in ["nf4", "fp4"]:
            config.quantization_config = QuantizationConfig()
        with device_guard(args.device):
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                config=config,
                low_cpu_mem_usage=True,
            )
            logger.info("load model done")
            model = LoRAModel.from_pretrained(model=model, lora_path=args.lora_path, lora_config=lora_config)
            logger.info("load lora model done")
        if quant_config.weight_quantize_algo in ["weight_only_int8"]:
            model.config.quantization_config = QuantizationConfig()
        model.eval()
        model_state_dict = model.model.state_dict()
        if quant_config.weight_quantize_algo in ["nf4", "fp4", "weight_only_int8"]:
            for name, layer in model.model.named_sublayers():
                if isinstance(layer, paddle.nn.Linear) or isinstance(layer, QuantizationLinear):
                    weight_process(name, quant_config, lora_config, model_state_dict, args.device)

        lora_name_list = []
        for key in model_state_dict.keys():
            if "lora_A" in key:
                lora_name_list.append(key[:-7])
        for name in lora_name_list:
            lora_process(name, lora_config, model_state_dict, args.device)

    logger.info("Begin to save merged model")
    if args.safe_serialization:
        model.model.save_pretrained(
            args.output_path, state_dict=model_state_dict, safe_serialization=args.safe_serialization
        )
    else:
        model.model.save_pretrained(args.output_path, state_dict=model_state_dict, max_shard_size="100GB")


if __name__ == "__main__":
    merge()
