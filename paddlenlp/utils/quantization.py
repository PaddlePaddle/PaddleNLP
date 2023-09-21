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
from paddle.nn.quant import llm_int8_linear, weight_only_linear, weight_quantize

from .log import logger

QuantDtypeMapping = {
    "weight_only_int8": "int8",
    "weight_only_int4": "int4",
    "llm.int8": "int8",
}


class QuantizationLinear(nn.Layer):
    def __init__(
        self, in_features, out_features, quant_algo, dtype, weight_attr=None, scale_attr=None, bias_attr=None, **kwargs
    ):
        super().__init__()

        self.quant_algo = quant_algo
        self.quant_dtype = QuantDtypeMapping[self.quant_algo]
        self._dtype = dtype
        if self.quant_algo == "llm.int8":
            self.llm_int8_threshold = kwargs.pop("llm_int8_threshold", 6.0)

        # PaddlePaddle dosen't support Int4 data type, one Int8 data represents two Int4 data.
        self.quant_weight = self.create_parameter(
            shape=[in_features // 2, out_features] if self.quant_dtype == "int4" else [in_features, out_features],
            attr=weight_attr if weight_attr else paddle.nn.initializer.Constant(value=0),
            dtype="int8",
            is_bias=False,
        )
        self.quant_scale = self.create_parameter(
            shape=[in_features],
            attr=scale_attr,
            dtype="float32",
            is_bias=False,
        )

        if bias_attr is False:
            self.bias = None
        else:
            self.bias = self.create_parameter(
                shape=[out_features],
                attr=bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )

    def forward(self, x):
        with paddle.amp.auto_cast(enable=False):
            if "weight_only" in self.quant_algo:
                out = weight_only_linear(x, self.quant_weight, self.bias, self.quant_scale, self.quant_dtype)
            else:
                out = llm_int8_linear(x, self.quant_weight, self.bias, self.quant_scale, self.llm_int8_threshold)
        return out


def replace_with_quantization_linear(model, quant_algo, name_prefix="", **kwargs):
    quantization_linear_list = []
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            if child.bias is None:
                bias_attr = False
            else:
                bias_attr = None

            model._sub_layers[name] = QuantizationLinear(
                child.weight.shape[0], child.weight.shape[1], quant_algo, child._dtype, bias_attr=bias_attr, **kwargs
            )
            del child
            quantization_linear_list.append(name_prefix + name)
        else:
            quantization_linear_list += replace_with_quantization_linear(
                child, quant_algo, name_prefix + name + ".", **kwargs
            )

    gc.collect()
    return quantization_linear_list


def convert_to_quantize_state_dict(state_dict, quantization_linear_list, quant_algo, dtype):
    for name in quantization_linear_list:
        weight_name = name + ".weight"
        quant_weight_name = name + ".quant_weight"
        quant_scale_name = name + ".quant_scale"

        if quant_weight_name in state_dict and quant_scale_name in state_dict:
            if state_dict[quant_weight_name].dtype != paddle.int8:
                raise ValueError(
                    f"{quant_weight_name} should be {paddle.int8} in state_dict but received dtype {state_dict[quant_weight_name].dtype}"
                )
            if state_dict[quant_scale_name].dtype != paddle.float32:
                raise ValueError(
                    f"{quant_scale_name} should be {paddle.float32} in state_dict but received dtype {state_dict[quant_scale_name].dtype}"
                )
        elif weight_name in state_dict:
            target_weight = state_dict.pop(weight_name).cast(dtype)
            quant_weight, quant_scale = weight_quantize(target_weight, quant_algo)
            state_dict[quant_weight_name] = quant_weight
            state_dict[quant_scale_name] = quant_scale
            del target_weight
        gc.collect()
    return state_dict


def update_loaded_state_dict_keys(state_dict, quantization_linear_list):
    for name in quantization_linear_list:
        weight_name = name + ".weight"
        quant_weight_name = name + ".quant_weight"
        quant_scale_name = name + ".quant_scale"

        if quant_weight_name in state_dict and quant_scale_name in state_dict:
            continue
        elif weight_name in state_dict:
            state_dict.remove(weight_name)
            state_dict.append(quant_weight_name)
            state_dict.append(quant_scale_name)
        else:
            logger.warning(
                f"Cannot find {weight_name} in state_dict or {quant_weight_name}  and {quant_scale_name} in state_dict"
            )

    return state_dict
