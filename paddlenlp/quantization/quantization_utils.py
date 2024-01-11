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

import gc

import paddle
import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddle.nn.quant import weight_quantize

from ..utils.log import logger
from .quantization_linear import (
    ColumnParallelQuantizationLinear,
    QuantizationLinear,
    RowParallelQuantizationLinear,
)

try:
    from .qlora import qlora_weight_quantize
except:
    qlora_weight_quantize = None


def replace_with_quantization_linear(model, quantization_config, name_prefix="", llm_int8_threshold=6.0):
    quantization_linear_list = []
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            if child.bias is None:
                bias_attr = False
            else:
                bias_attr = None

            model._sub_layers[name] = QuantizationLinear(
                child.weight.shape[0],
                child.weight.shape[1],
                quantization_config.weight_quantize_algo,
                child._dtype,
                bias_attr=bias_attr,
                llm_int8_threshold=llm_int8_threshold,
                block_size=quantization_config.weight_blocksize,
                double_quant_block_size=quantization_config.weight_double_quant_block_size,
                double_quant=quantization_config.weight_double_quant,
            )
            del child
            quantization_linear_list.append(name_prefix + name)
        elif isinstance(child, ColumnParallelLinear):
            if child.bias is None:
                bias_attr = False
            else:
                bias_attr = None
            model._sub_layers[name] = ColumnParallelQuantizationLinear(
                child.weight.shape[0],
                child.weight.shape[1] * child.world_size,
                quantization_config.weight_quantize_algo,
                child._dtype,
                bias_attr=bias_attr,
                gather_output=child.gather_output,
                llm_int8_threshold=llm_int8_threshold,
            )
            del child
            quantization_linear_list.append(name_prefix + name)
        elif isinstance(child, RowParallelLinear):
            if child.bias is None:
                bias_attr = False
            else:
                bias_attr = None
            model._sub_layers[name] = RowParallelQuantizationLinear(
                child.weight.shape[0] * child.world_size,
                child.weight.shape[1],
                quantization_config.weight_quantize_algo,
                child._dtype,
                bias_attr=bias_attr,
                input_is_parallel=child.input_is_parallel,
                llm_int8_threshold=llm_int8_threshold,
            )
            del child
            quantization_linear_list.append(name_prefix + name)
        else:
            quantization_linear_list += replace_with_quantization_linear(
                child, quantization_config, name_prefix + name + ".", llm_int8_threshold
            )

    gc.collect()
    return quantization_linear_list


def convert_to_quantize_state_dict_with_check(state_dict, quantization_linear_list, quant_algo, dtype):
    for name in quantization_linear_list:
        weight_name = name + ".weight"
        quant_weight_name = name + ".quant_weight"
        quant_scale_name = name + ".quant_scale"

        if quant_weight_name in state_dict and quant_scale_name in state_dict:
            if state_dict[quant_weight_name].dtype != paddle.int8:
                raise ValueError(
                    f"{quant_weight_name} should be {paddle.int8} in state_dict but received dtype {state_dict[quant_weight_name].dtype}."
                )
            if (
                state_dict[quant_scale_name].dtype != paddle.float16
                and state_dict[quant_scale_name].dtype != paddle.bfloat16
            ):
                raise ValueError(
                    f"{quant_scale_name} should be {paddle.float16} or {paddle.bfloat16} in state_dict but received dtype {state_dict[quant_scale_name].dtype}."
                )
        elif weight_name in state_dict:
            target_weight = state_dict.pop(weight_name).cast(dtype)
            quant_weight, quant_scale = weight_quantize(target_weight, quant_algo)
            state_dict[quant_weight_name] = quant_weight
            state_dict[quant_scale_name] = quant_scale
            del target_weight
        gc.collect()
    return state_dict


def convert_to_quantize_state_dict_without_check(state_dict, quantization_linear_list, quantization_config, dtype):
    if qlora_weight_quantize is None:
        raise ImportError(
            "Please run the following commands to install qlora related package first: \n"
            "1) git clone https://github.com/PaddlePaddle/PaddleSlim \n"
            "2) cd PaddleSlim \n"
            "3) python ./csrc/setup_cuda.py install"
        )
    for name in quantization_linear_list:
        weight_name = name + ".weight"
        if weight_name in state_dict:
            target_weight = state_dict.pop(weight_name).cast(dtype).cuda()
            qlora_state_dict = qlora_weight_quantize(
                weight=target_weight,
                quant_algo=quantization_config.weight_quantize_algo,
                double_quant=quantization_config.weight_double_quant,
                block_size=quantization_config.weight_blocksize,
                double_quant_block_size=quantization_config.weight_double_quant_block_size,
                linear_name=name,
                return_dict=True,
            )
            state_dict.update(qlora_state_dict)
            del target_weight
            gc.collect()
            paddle.device.cuda.empty_cache()
    return state_dict


def convert_to_quantize_state_dict(state_dict, quantization_linear_list, quantization_config, dtype):
    if quantization_config.weight_quantize_algo in ["weight_only_int8", "weight_only_int4", "llm.int8"]:
        return convert_to_quantize_state_dict_with_check(
            state_dict, quantization_linear_list, quantization_config.weight_quantize_algo, dtype
        )
    elif quantization_config.weight_quantize_algo in ["fp4", "nf4"]:
        return convert_to_quantize_state_dict_without_check(
            state_dict, quantization_linear_list, quantization_config, dtype
        )
    else:
        raise NotImplementedError(
            f"Please check the quantization_config.weight_quantize_algo: {quantization_config.weight_quantize_algo}"
        )


def update_loaded_state_dict_keys(state_dict, quantization_linear_list, quantization_config):
    for name in quantization_linear_list:
        weight_name = name + ".weight"
        quant_weight_name = name + ".quant_weight"
        quant_scale_name = name + ".quant_scale"
        qquant_scale_name = name + ".qquant_scale"
        double_quant_scale_name = name + ".double_quant_scale"
        quant_sacle_offset_name = name + ".quant_sacle_offset"

        if quant_weight_name in state_dict and quant_scale_name in state_dict:
            continue
        elif weight_name in state_dict:
            state_dict.remove(weight_name)
            state_dict.append(quant_weight_name)
            if quantization_config.weight_double_quant:
                state_dict.append(qquant_scale_name)
                state_dict.append(double_quant_scale_name)
                state_dict.append(quant_sacle_offset_name)
            else:
                state_dict.append(quant_scale_name)
        else:
            logger.warning(
                f"Cannot find {weight_name} in state_dict or {quant_weight_name}  and {quant_scale_name} in state_dict"
            )

    return state_dict
