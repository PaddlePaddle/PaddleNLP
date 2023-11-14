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
from paddle.distributed.fleet.base import topology as tp
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    get_rng_state_tracker,
)
from paddle.nn.quant import llm_int8_linear, weight_only_linear, weight_quantize

from .log import logger

try:
    from .qlora import qlora_weight_linear, qlora_weight_quantize
except:
    qlora_weight_linear = None
    qlora_weight_quantize = None


QuantMapping = {
    # (quant_dtype, quant_weight_dtype, quant_weight_bit)
    "weight_only_int8": ("int8", "int8", 8),
    "weight_only_int4": ("int4", "int8", 4),
    "llm.int8": ("int8", "int8", 8),
    "fp4": ("fp4", "uint8", 4),
    "nf4": ("nf4", "uint8", 4),
}


class QuantizationLinear(nn.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        quant_algo,
        dtype,
        weight_attr=None,
        scale_attr=None,
        bias_attr=None,
        block_size=64,
        double_quant_block_size=256,
        double_quant=False,
        qquant_scale_attr=None,
        double_quant_scale_attr=None,
        quant_sacle_offset_attr=None,
        quant_scale_attr=None,
        llm_int8_threshold=6.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_algo = quant_algo
        self.quant_dtype, self.quant_weight_dtype, self.quant_weight_bit = QuantMapping[self.quant_algo]
        self._dtype = dtype
        self.llm_int8_threshold = llm_int8_threshold
        self.block_size = block_size
        self.double_quant_block_size = double_quant_block_size
        self.double_quant = double_quant

        # PaddlePaddle dosen't support 4bit data type, one 8bit data represents two 4bit data.
        # paddle.nn.quant.weight_quantize will transpose in_features and out_features.
        self.quant_weight = self.create_parameter(
            shape=[out_features // 2, in_features] if self.quant_weight_bit == "4" else [out_features, in_features],
            attr=weight_attr if weight_attr else paddle.nn.initializer.Constant(value=0),
            dtype=self.quant_weight_dtype,
            is_bias=False,
        )
        if self.quant_algo in ["weight_only_int8", "weight_only_int4", "llm.int8"]:
            self.quant_scale = self.create_parameter(
                shape=[out_features],
                attr=scale_attr,
                dtype="float32",
                is_bias=False,
            )
        if self.quant_algo in ["fp4", "nf4"]:
            if qlora_weight_linear is None:
                raise ImportError(
                    "Please run the following commands to install: qlora related package first\n"
                    "1) git clone https://github.com/PaddlePaddle/PaddleSlim \n"
                    "2) cd PaddleSlim/csrc \n"
                    "3) python ./setup_cuda.py install"
                )
            if self.double_quant:
                # quantized quant_scale
                self.qquant_scale = self.create_parameter(
                    shape=[in_features * out_features // self.block_size],
                    attr=qquant_scale_attr if qquant_scale_attr else paddle.nn.initializer.Constant(value=0),
                    dtype="uint8",
                    is_bias=False,
                )
                # double quant_scale: quant_scale of quantized quant_scale
                self.double_quant_scale = self.create_parameter(
                    shape=[in_features * out_features // self.block_size // self.double_quant_block_size],
                    attr=double_quant_scale_attr,
                    dtype="float32",
                    is_bias=False,
                )
                self.quant_sacle_offset = self.create_parameter(
                    shape=[],
                    attr=quant_sacle_offset_attr,
                    dtype="float32",
                    is_bias=False,
                )
            else:
                self.quant_scale = self.create_parameter(
                    shape=[in_features * out_features // self.block_size],
                    attr=quant_scale_attr if quant_scale_attr else paddle.nn.initializer.Constant(value=0),
                    dtype="uint8",
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
            if self.quant_algo in ["weight_only_int8", "weight_only_int4"]:
                out = weight_only_linear(x, self.quant_weight, self.bias, self.quant_scale, self.quant_dtype)
            elif self.quant_algo in ["llm.int8"]:
                out = llm_int8_linear(x, self.quant_weight, self.bias, self.quant_scale, self.llm_int8_threshold)
            elif self.quant_algo in ["fp4", "nf4"]:
                out = qlora_weight_linear(
                    x,
                    self.quant_weight.reshape([-1, 1]),
                    self.quant_algo,
                    (self.qquant_scale, self.double_quant_scale, self.quant_sacle_offset)
                    if self.double_quant
                    else self.quant_scale,
                    self.double_quant,
                    self.block_size,
                    self.double_quant_block_size,
                    self.bias,
                )
        return out


class ColumnParallelQuantizationLinear(nn.Layer):
    """Quantization Linear layer with mp parallelized(column).
    The code implementation refers to paddle.distributed.fleet.meta_parallel.ColumnParallelLinear.
    https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/distributed/fleet/layers/mpu/mp_layers.py#L310
    Different from ColumnParallelLinear, this class keeps weight in INT8/INT4 with quant scale, and supports matrix
    multiplication(weight_only_linear/llm_int8_linear) for input tensor(fp16/bf16) and quantized weight(INT8/INT4)
    and bias addition if provided.
    Notice: quantized weight shape is transposed of weight shape in ColumnParallelLinear.
    """

    def __init__(
        self,
        in_features,
        out_features,
        quant_algo,
        dtype,
        weight_attr=None,
        scale_attr=None,
        bias_attr=None,
        gather_output=True,
        mp_group=None,
        llm_int8_threshold=6.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_algo = quant_algo
        self.quant_dtype, self.quant_weight_dtype, self.quant_weight_bit = QuantMapping[self.quant_algo]
        self._dtype = dtype
        self.llm_int8_threshold = llm_int8_threshold

        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group() if mp_group is None else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size() if mp_group is None else mp_group.nranks
        )
        self.is_mp = self.world_size > 1
        self.gather_output = gather_output
        self.output_size_per_partition = out_features // self.world_size

        # PaddlePaddle dosen't support Int4 data type, one Int8 data represents two Int4 data.
        if self.is_mp and paddle.in_dynamic_mode():
            with get_rng_state_tracker().rng_state():
                self.quant_weight = self.create_parameter(
                    shape=[self.output_size_per_partition // 2, in_features]
                    if self.quant_dtype == "int4"
                    else [self.output_size_per_partition, in_features],
                    attr=weight_attr if weight_attr else paddle.nn.initializer.Constant(value=0),
                    dtype="int8",
                    is_bias=False,
                )
        else:
            self.quant_weight = self.create_parameter(
                shape=[self.output_size_per_partition // 2, in_features]
                if self.quant_dtype == "int4"
                else [self.output_size_per_partition, in_features],
                attr=weight_attr if weight_attr else paddle.nn.initializer.Constant(value=0),
                dtype="int8",
                is_bias=False,
            )

        self.quant_weight.is_distributed = True if self.is_mp else False
        if self.quant_weight.is_distributed:
            self.quant_weight.split_axis = 0

        self.quant_scale = self.create_parameter(
            shape=[self.output_size_per_partition],
            attr=scale_attr,
            dtype="float32",
            is_bias=False,
        )
        self.quant_scale.is_distributed = True if self.is_mp else False
        if self.quant_scale.is_distributed:
            self.quant_scale.split_axis = 0

        if bias_attr is False:
            self.bias = None
        else:
            self.bias = self.create_parameter(
                shape=[self.output_size_per_partition],
                attr=bias_attr if bias_attr else paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype,
                is_bias=True,
            )
            self.bias.is_distributed = True if self.is_mp else False
            if self.bias.is_distributed:
                self.bias.split_axis = 0

    def forward(self, x):
        if self.is_mp:
            input_parallel = mp_ops._c_identity(x, group=self.model_parallel_group)
        else:
            input_parallel = x

        with paddle.amp.auto_cast(enable=False):
            if "weight_only" in self.quant_algo:
                output_parallel = weight_only_linear(
                    input_parallel, self.quant_weight, self.bias, self.quant_scale, self.quant_dtype
                )
            else:
                output_parallel = llm_int8_linear(
                    input_parallel, self.quant_weight, self.bias, self.quant_scale, self.llm_int8_threshold
                )

        if self.gather_output and self.is_mp:
            output = mp_ops._c_concat(output_parallel, group=self.model_parallel_group)
        else:
            output = output_parallel
        return output


class RowParallelQuantizationLinear(nn.Layer):
    """Quantization Linear layer with mp parallelized(row).
    The code implementation refers to paddle.distributed.fleet.meta_parallel.RowParallelLinear.
    https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/distributed/fleet/layers/mpu/mp_layers.py#L517
    Different from RowParallelLinear, this class keeps weight in INT8/INT4 with quant scale, and supports matrix
    multiplication(weight_only_linear/llm_int8_linear) for input tensor(fp16/bf16) and quantized weight(INT8/INT4)
    and bias addition if provided.
    Notice: quantized weight shape is transposed of weight shape in RowParallelLinear.
    """

    def __init__(
        self,
        in_features,
        out_features,
        quant_algo,
        dtype,
        weight_attr=None,
        scale_attr=None,
        bias_attr=None,
        input_is_parallel=False,
        mp_group=None,
        llm_int8_threshold=6.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_algo = quant_algo
        self.quant_dtype, self.quant_weight_dtype, self.quant_weight_bit = QuantMapping[self.quant_algo]
        self._dtype = dtype
        self.llm_int8_threshold = llm_int8_threshold

        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group() if mp_group is None else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size() if mp_group is None else mp_group.nranks
        )
        self.rank = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_rank() if mp_group is None else mp_group.rank
        self.is_mp = self.world_size > 1
        self.input_is_parallel = input_is_parallel
        self.input_size_per_partition = in_features // self.world_size

        # PaddlePaddle dosen't support Int4 data type, one Int8 data represents two Int4 data.
        # paddle.nn.quant.weight_quantize will transpose in_features and out_features.
        if self.is_mp and paddle.in_dynamic_mode():
            with get_rng_state_tracker().rng_state():
                self.quant_weight = self.create_parameter(
                    shape=[out_features // 2, self.input_size_per_partition]
                    if self.quant_dtype == "int4"
                    else [out_features, self.input_size_per_partition],
                    attr=weight_attr if weight_attr else paddle.nn.initializer.Constant(value=0),
                    dtype="int8",
                    is_bias=False,
                )
        else:
            self.quant_weight = self.create_parameter(
                shape=[out_features // 2, self.input_size_per_partition]
                if self.quant_dtype == "int4"
                else [out_features, self.input_size_per_partition],
                attr=weight_attr if weight_attr else paddle.nn.initializer.Constant(value=0),
                dtype="int8",
                is_bias=False,
            )

        self.quant_weight.is_distributed = True if self.is_mp else False
        if self.quant_weight.is_distributed:
            self.quant_weight.split_axis = 1

        self.quant_scale = self.create_parameter(
            shape=[out_features],
            attr=scale_attr,
            dtype="float32",
            is_bias=False,
        )
        self.quant_scale.is_distributed = True if self.is_mp else False
        if self.quant_scale.is_distributed:
            self.quant_scale.split_axis = 0

        if bias_attr is False:
            self.bias = None
        else:
            self.bias = self.create_parameter(
                shape=[out_features],
                attr=bias_attr if bias_attr else paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype,
                is_bias=True,
            )

    def forward(self, x):
        if self.input_is_parallel or (not self.is_mp):
            input_parallel = x
        else:
            # split last dim
            input_parallel = mp_ops._c_split(x, group=self.model_parallel_group)

        if self.is_mp:
            with paddle.amp.auto_cast(enable=False):
                if "weight_only" in self.quant_algo:
                    output_parallel = weight_only_linear(
                        input_parallel, self.quant_weight, None, self.quant_scale, self.quant_dtype
                    )
                else:
                    output_parallel = llm_int8_linear(
                        input_parallel, self.quant_weight, None, self.quant_scale, self.llm_int8_threshold
                    )

            output_ = mp_ops._mp_allreduce(
                output_parallel,
                group=self.model_parallel_group,
                use_calc_stream=True,
                use_model_parallel=True,
            )
            output = output_ + self.bias if self.bias is not None else output_

        else:
            with paddle.amp.auto_cast(enable=False):
                if "weight_only" in self.quant_algo:
                    output = weight_only_linear(
                        input_parallel, self.quant_weight, self.bias, self.quant_scale, self.quant_dtype
                    )
                else:
                    output = llm_int8_linear(
                        input_parallel, self.quant_weight, self.bias, self.quant_scale, self.llm_int8_threshold
                    )
        return output


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
            if state_dict[quant_scale_name].dtype != paddle.float32:
                raise ValueError(
                    f"{quant_scale_name} should be {paddle.float32} in state_dict but received dtype {state_dict[quant_scale_name].dtype}."
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
    if qlora_weight_linear is None:
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
                target_weight,
                quantization_config.weight_quantize_algo,
                double_quant=quantization_config.weight_double_quant,
                block_size=quantization_config.weight_blocksize,
                double_quant_block_size=quantization_config.weight_double_quant_block_size,
                linear_name=name,
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
