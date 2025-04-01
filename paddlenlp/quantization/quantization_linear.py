# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
from paddle.distributed.fleet.base import topology as tp
from paddle.distributed.fleet.layers.mpu import mp_ops

try:
    from paddle.nn.quant import llm_int8_linear, weight_only_linear
except:
    llm_int8_linear = None
    weight_only_linear = None
try:
    from .qlora import qlora_weight_linear
except:
    qlora_weight_linear = None


QuantMapping = {
    # (quant_dtype, quant_weight_bit)
    "weight_only_int8": ("int8", 8),
    "weight_only_int4": ("int4", 4),
    "llm.int8": ("int8", 8),
    "fp4": ("fp4", 4),
    "nf4": ("nf4", 4),
}


class QuantizationLinear(nn.Layer):
    """Quantization Linear layer."""

    def __init__(
        self,
        in_features,
        out_features,
        quantization_config,
        weight_quantize_algo,
        dtype,
        bias_attr=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantization_config = quantization_config
        self.weight_quantize_algo = weight_quantize_algo
        self._dtype = dtype
        self.quant_dtype, self.quant_weight_bit = QuantMapping[self.weight_quantize_algo]

        # PaddlePaddle dosen't support 4bit data type, one 8bit data represents two 4bit data.
        # paddle.nn.quant.weight_quantize will transpose in_features and out_features.
        if self.weight_quantize_algo in ["weight_only_int8", "weight_only_int4", "llm.int8"]:
            self.quant_weight = self.create_parameter(
                shape=[out_features // 2, in_features] if self.quant_weight_bit == 4 else [out_features, in_features],
                dtype="int8",
                is_bias=False,
            )
            if self.quantization_config.group_size == -1:
                self.quant_scale = self.create_parameter(
                    shape=[out_features],
                    dtype=self._dtype,
                    is_bias=False,
                )
            else:
                # TODO(lugimzzz): support groupwise in next PR
                raise NotImplementedError("Not yet support grouwise weightonly quantization.")
        elif self.weight_quantize_algo in ["fp4", "nf4"]:
            if qlora_weight_linear is None:
                raise ImportError(
                    "Please run the following commands to install: qlora related package first\n"
                    "1) git clone https://github.com/PaddlePaddle/PaddleSlim \n"
                    "2) cd PaddleSlim && pip install -e .\n"
                    "3) cd csrc &&  python ./setup_cuda.py install"
                )
            self.quant_weight = self.create_parameter(
                shape=[out_features * in_features // 2, 1],
                attr=paddle.nn.initializer.Constant(value=0),
                dtype="uint8",
                is_bias=False,
            )
            if self.quantization_config.qlora_weight_double_quant:
                # quantized quant_scale
                self.qquant_scale = self.create_parameter(
                    shape=[in_features * out_features // self.quantization_config.qlora_weight_blocksize],
                    dtype="uint8",
                    is_bias=False,
                )
                # double quant_scale: quant_scale of quantized quant_scale
                self.double_quant_scale = self.create_parameter(
                    shape=[
                        in_features
                        * out_features
                        // self.quantization_config.qlora_weight_blocksize
                        // self.quantization_config.qlora_weight_double_quant_block_size
                    ],
                    dtype="float32",
                    is_bias=False,
                )
                self.quant_scale_offset = self.create_parameter(
                    shape=[],
                    dtype="float32",
                    is_bias=False,
                )
            else:
                self.quant_scale = self.create_parameter(
                    shape=[in_features * out_features // self.quantization_config.qlora_weight_blocksize],
                    dtype="float32",
                    is_bias=False,
                )
        else:
            raise NotImplementedError(f"Not yet support weight_quantize_algo: {self.weight_quantize_algo}")
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
            if self.weight_quantize_algo in ["weight_only_int8", "weight_only_int4"]:
                out = weight_only_linear(
                    x=x,
                    weight=self.quant_weight,
                    bias=self.bias,
                    weight_scale=self.quant_scale,
                    weight_dtype=self.quant_dtype,
                    group_size=self.quantization_config.group_size,
                )
            elif self.weight_quantize_algo in ["llm.int8"]:
                out = llm_int8_linear(
                    x, self.quant_weight, self.bias, self.quant_scale, self.self.quantization_config.llm_int8_threshold
                )
            elif self.weight_quantize_algo in ["fp4", "nf4"]:
                out = qlora_weight_linear(
                    x=x,
                    quant_weight=self.quant_weight,
                    dtype=self._dtype,
                    state=(self.qquant_scale, self.double_quant_scale, self.quant_scale_offset)
                    if self.quantization_config.qlora_weight_double_quant
                    else self.quant_scale,
                    quant_algo=self.weight_quantize_algo,
                    double_quant=self.quantization_config.qlora_weight_double_quant,
                    block_size=self.quantization_config.qlora_weight_blocksize,
                    double_quant_block_size=self.quantization_config.qlora_weight_double_quant_block_size,
                    bias=self.bias,
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
        output_size_per_partition,
        quantization_config,
        weight_quantize_algo,
        dtype,
        bias_attr=None,
        gather_output=True,
        mp_skip_c_identity=False,
        mp_group=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.output_size_per_partition = output_size_per_partition
        self.weight_quantize_algo = weight_quantize_algo
        self.quantization_config = quantization_config
        self._dtype = dtype
        self.mp_skip_c_identity = mp_skip_c_identity
        self.quant_dtype, self.quant_weight_bit = QuantMapping[self.weight_quantize_algo]

        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group() if mp_group is None else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size() if mp_group is None else mp_group.nranks
        )
        self.is_mp = self.world_size > 1
        self.gather_output = gather_output

        # PaddlePaddle dosen't support Int4 data type, one Int8 data represents two Int4 data.
        if self.weight_quantize_algo in ["weight_only_int8", "weight_only_int4", "llm.int8"]:
            self.quant_weight = self.create_parameter(
                shape=[self.output_size_per_partition // 2, in_features]
                if self.quant_dtype == "int4"
                else [self.output_size_per_partition, in_features],
                dtype="int8",
                is_bias=False,
            )
            self.quant_weight.is_distributed = True if self.is_mp else False
            if self.quant_weight.is_distributed:
                self.quant_weight.split_axis = 0

            if self.quantization_config.group_size == -1:
                self.quant_scale = self.create_parameter(
                    shape=[self.output_size_per_partition],
                    dtype=self._dtype,
                    is_bias=False,
                )
                self.quant_scale.is_distributed = True if self.is_mp else False
                if self.quant_scale.is_distributed:
                    self.quant_scale.split_axis = 0
            else:
                # TODO(lugimzzz): support groupwise in next PR
                raise NotImplementedError("Not yet support grouwise weightonly quantization.")
        else:
            raise NotImplementedError(f"Not yet support weight_quantize_algo: {self.weight_quantize_algo}")
        if bias_attr is False:
            self.bias = None
        else:
            self.bias = self.create_parameter(
                shape=[self.output_size_per_partition],
                attr=bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
            self.bias.is_distributed = True if self.is_mp else False
            if self.bias.is_distributed:
                self.bias.split_axis = 0

    def forward(self, x):
        if self.is_mp:
            input_parallel = mp_ops._c_identity(
                x,
                group=self.model_parallel_group,
                skip_c_identity_dynamic=self.mp_skip_c_identity,
            )
        else:
            input_parallel = x

        # with paddle.amp.auto_cast(enable=False):
        if self.weight_quantize_algo in ["weight_only_int8", "weight_only_int4"]:
            output_parallel = weight_only_linear(
                x=input_parallel,
                weight=self.quant_weight,
                bias=self.bias,
                weight_scale=self.quant_scale,
                weight_dtype=self.quant_dtype,
                group_size=self.quantization_config.group_size,
            )
        elif self.weight_quantize_algo in ["llm.int8"]:
            output_parallel = llm_int8_linear(
                input_parallel,
                self.quant_weight,
                self.bias,
                self.quant_scale,
                self.self.quantization_config.llm_int8_threshold,
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
        input_size_per_partition,
        out_features,
        quantization_config,
        weight_quantize_algo,
        dtype,
        bias_attr=None,
        input_is_parallel=False,
        mp_skip_c_identity=False,
        mp_group=None,
    ):
        super().__init__()
        self.input_size_per_partition = input_size_per_partition
        self.out_features = out_features
        self.quantization_config = quantization_config
        self.weight_quantize_algo = weight_quantize_algo
        self._dtype = dtype
        self.mp_skip_c_identity = mp_skip_c_identity
        self.quant_dtype, self.quant_weight_bit = QuantMapping[self.weight_quantize_algo]

        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group() if mp_group is None else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size() if mp_group is None else mp_group.nranks
        )
        self.is_mp = self.world_size > 1
        self.input_is_parallel = input_is_parallel

        # PaddlePaddle dosen't support Int4 data type, one Int8 data represents two Int4 data.
        # paddle.nn.quant.weight_quantize will transpose in_features and out_features.
        if self.weight_quantize_algo in ["weight_only_int8", "weight_only_int4", "llm.int8"]:
            self.quant_weight = self.create_parameter(
                shape=[out_features // 2, self.input_size_per_partition]
                if self.quant_dtype == "int4"
                else [out_features, self.input_size_per_partition],
                dtype="int8",
                is_bias=False,
            )
            self.quant_weight.is_distributed = True if self.is_mp else False
            if self.quant_weight.is_distributed:
                self.quant_weight.split_axis = 1

            if self.quantization_config.group_size == -1:
                self.quant_scale = self.create_parameter(
                    shape=[out_features],
                    dtype=self._dtype,
                    is_bias=False,
                )
                self.quant_scale.is_distributed = True if self.is_mp else False
                if self.quant_scale.is_distributed:
                    self.quant_scale.split_axis = 0
            else:
                # TODO(lugimzzz): support groupwise in next PR
                raise NotImplementedError("Not yet support grouwise weightonly quantization.")
        else:
            raise NotImplementedError(f"Not yet support weight_quantize_algo: {self.weight_quantize_algo}")

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
        if self.input_is_parallel or (not self.is_mp):
            input_parallel = x
        else:
            # split last dim
            input_parallel = mp_ops._c_split(x, group=self.model_parallel_group)

        # with paddle.amp.auto_cast(enable=False):
        if self.is_mp:
            if self.weight_quantize_algo in ["weight_only_int8", "weight_only_int4"]:
                output_parallel = weight_only_linear(
                    x=input_parallel,
                    weight=self.quant_weight,
                    bias=None,
                    weight_scale=self.quant_scale,
                    weight_dtype=self.quant_dtype,
                    group_size=self.quantization_config.group_size,
                )
            elif self.weight_quantize_algo in ["llm.int8"]:
                output_parallel = llm_int8_linear(
                    input_parallel,
                    self.quant_weight,
                    None,
                    self.quant_scale,
                    self.quantization_config.llm_int8_threshold,
                )
            output_ = mp_ops._mp_allreduce(
                output_parallel,
                group=self.model_parallel_group,
                use_calc_stream=True,
                use_model_parallel=True,
                skip_c_identity_dynamic=self.mp_skip_c_identity,
            )
            output = output_ + self.bias if self.bias is not None else output_
        else:
            if self.weight_quantize_algo in ["weight_only_int8", "weight_only_int4"]:
                output = weight_only_linear(
                    x=input_parallel,
                    weight=self.quant_weight,
                    bias=self.bias,
                    weight_scale=self.quant_scale,
                    weight_dtype=self.quant_dtype,
                    group_size=self.quantization_config.group_size,
                )
            elif self.weight_quantize_algo in ["llm.int8"]:
                output = llm_int8_linear(
                    input_parallel, self.quant_weight, self.bias, self.quant_scale, self.llm_int8_threshold
                )

        return output
