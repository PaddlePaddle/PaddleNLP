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
from paddle.autograd import PyLayer
from paddle.distributed.fleet.base import topology as tp
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    AllGatherOp,
    ReduceScatterOp,
)
from paddle.nn.quant import llm_int8_linear, weight_dequantize, weight_only_linear

from paddlenlp.utils import infohub

try:
    from .qlora import qlora_weight_dequantize, qlora_weight_linear
except:
    qlora_weight_linear = None
    qlora_weight_dequantize = None

QuantMapping = {
    # (quant_dtype, quant_weight_bit)
    "weight_only_int8": ("int8", 8),
    "weight_only_int4": ("int4", 4),
    "llm.int8": ("int8", 8),
    "fp4": ("fp4", 4),
    "nf4": ("nf4", 4),
    "a8w8linear": ("int8", 8),
}


def random_hadamard(n, dtype):
    A = paddle.randint(low=0, high=2, shape=[n, n]).astype("float32") * 2 - 1
    Q, _ = paddle.linalg.qr(A)
    return Q.astype(dtype)


def quantize_tensorwise(x, apply_hadamard=False, qmax=127, qmin=-128):
    if apply_hadamard:
        target_x = x @ infohub.hadamard[x.shape[-1]]
    else:
        target_x = x.clone()
    scale = paddle.max(paddle.abs(target_x)) / qmax
    x_int8 = paddle.clip((target_x / scale).round(), qmin, qmax).astype("int8")
    return x_int8, scale


def dequantize_tensorwise(x_int8, scale, apply_hadamard=False):
    x = x_int8.astype(scale.dtype) * scale
    if apply_hadamard:
        x = x @ infohub.hadamard[x.shape[-1]].T
    return x


def quantize_channelwise(w, apply_hadamard=False, qmax=127, qmin=-128):
    if apply_hadamard:
        if getattr(infohub, "hadamard") is None:
            setattr(infohub, "hadamard", {})
        if w.shape[0] in infohub.hadamard:
            hadamard_matrix = infohub.hadamard[w.shape[0]]
        else:
            hadamard_matrix = random_hadamard(w.shape[0], w.dtype)
            infohub.hadamard[w.shape[0]] = hadamard_matrix
        w = hadamard_matrix.T @ w
    scale = paddle.max(paddle.abs(w), axis=0, keepdim=True) / qmax
    w_int8 = paddle.clip((w / scale).round(), qmin, qmax).astype("int8")
    return w_int8.T, scale.squeeze(0)


def dequantize_channelwise(w_int8, scale, apply_hadamard=False):
    w = w_int8.T.astype(scale.dtype) * scale
    if apply_hadamard:
        w = infohub.hadamard[w_int8.shape[1]] @ w
    return w


def a8w8_linear(x, w_int8, w_scale=None, bias=None, dtype=None, apply_hadamard=False):
    # if w_int8.dtype != paddle.int8:
    #     w_int8, w_scale = quantize_channelwise(w_int8, apply_hadamard)
    # if dtype is None:
    #     dtype = x.dtype
    x_int8, x_scale = quantize_tensorwise(x, apply_hadamard)
    out = paddle.matmul(x_int8, w_int8.T).astype(dtype) * x_scale * w_scale.unsqueeze(0)
    if bias is not None:
        out += bias
    return out


def quant_weight_forward(
    x,
    quant_weight,
    bias,
    quant_scale,
    quant_state,
    quant_dtype,
    quantization_config,
    weight_quantize_algo,
    dtype,
):
    if weight_quantize_algo in ["weight_only_int8", "weight_only_int4"]:
        output = weight_only_linear(
            x=x,
            weight=quant_weight,
            bias=bias,
            weight_scale=quant_scale,
            weight_dtype=quant_dtype,
            group_size=quantization_config.group_size,
        )
    elif weight_quantize_algo in ["llm.int8"]:
        output = llm_int8_linear(x, quant_weight, bias, quant_scale, quantization_config.llm_int8_threshold)
    elif weight_quantize_algo in ["fp4", "nf4"]:
        output = qlora_weight_linear(
            x=x,
            quant_weight=quant_weight,
            dtype=dtype,
            state=quant_state if quantization_config.qlora_weight_double_quant else quant_scale,
            quant_algo=weight_quantize_algo,
            double_quant=quantization_config.qlora_weight_double_quant,
            block_size=quantization_config.qlora_weight_blocksize,
            double_quant_block_size=quantization_config.qlora_weight_double_quant_block_size,
            bias=bias,
        )
    elif weight_quantize_algo in ["a8w8linear"]:
        output = a8w8_linear(
            x,
            quant_weight,
            w_scale=quant_scale,
            bias=bias,
            dtype=dtype,
            apply_hadamard=quantization_config.apply_hadamard,
        )
    return output


def dequant_weight(
    quant_weight,
    quantization_config,
    weight_quantize_algo,
    dtype,
    quant_scale,
    quant_state,
    input_shape,
):
    if weight_quantize_algo in ["weight_only_int8", "weight_only_int4", "llm.int8"]:
        quant_dequant_weight = weight_dequantize(
            x=quant_weight,
            scale=quant_scale,
            algo=weight_quantize_algo,
            out_dtype=dtype,
            group_size=quantization_config.group_size,
        )
    elif weight_quantize_algo in ["fp4", "nf4"]:
        quant_dequant_weight = (
            qlora_weight_dequantize(
                quant_weight=quant_weight,
                quant_algo=weight_quantize_algo,
                state=quant_state if quantization_config.qlora_weight_double_quant else quant_scale,
                double_quant=quantization_config.qlora_weight_double_quant,
                block_size=quantization_config.qlora_weight_blocksize,
                double_quant_block_size=quantization_config.qlora_weight_double_quant_block_size,
            )
            .reshape([input_shape[-1], -1])
            .cast(dtype)
        )
    elif weight_quantize_algo in ["a8w8linear"]:
        quant_dequant_weight = dequantize_channelwise(
            quant_weight, quant_scale, apply_hadamard=quantization_config.apply_hadamard
        )
    return quant_dequant_weight


class QuantizationLinearFunc(PyLayer):
    @staticmethod
    def forward(
        ctx,
        x,
        quant_weight,
        bias,
        quant_scale,
        quant_state,
        quant_dtype,
        quantization_config,
        weight_quantize_algo,
        dtype,
    ):

        output = quant_weight_forward(
            x=x,
            quant_weight=quant_weight,
            bias=bias,
            quant_scale=quant_scale,
            quant_state=quant_state,
            quant_dtype=quant_dtype,
            quantization_config=quantization_config,
            weight_quantize_algo=weight_quantize_algo,
            dtype=dtype,
        )
        ctx.quant_dtype = quant_dtype
        ctx.quantization_config = quantization_config
        ctx.weight_quantize_algo = weight_quantize_algo
        ctx.dtype = dtype
        if ctx.weight_quantize_algo in ["fp4", "nf4"] and ctx.quantization_config.qlora_weight_double_quant:
            qquant_scale, double_quant_scale, quant_scale_offset = quant_state
            ctx.save_for_backward(x, quant_weight, bias, qquant_scale, double_quant_scale, quant_scale_offset)
        else:
            ctx.save_for_backward(x, quant_weight, bias, quant_scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.weight_quantize_algo in ["fp4", "nf4"] and ctx.quantization_config.qlora_weight_double_quant:
            x, quant_weight, bias, qquant_scale, double_quant_scale, quant_scale_offset = ctx.saved_tensor()
            quant_state = (qquant_scale, double_quant_scale, quant_scale_offset)
            quant_scale = None
        else:
            x, quant_weight, bias, quant_scale = ctx.saved_tensor()
            quant_state = None

        qdq_weight = dequant_weight(
            quant_weight=quant_weight,
            quantization_config=ctx.quantization_config,
            weight_quantize_algo=ctx.weight_quantize_algo,
            dtype=ctx.dtype,
            quant_scale=quant_scale,
            quant_state=quant_state,
            input_shape=x.shape,
        )

        if not x.stop_gradient:
            input_grad = paddle.matmul(grad_output, qdq_weight.T)
        else:
            input_grad = None

        if not quant_weight.stop_gradient:
            weight_grad = paddle.einsum("bsh,bsd->hd", x, grad_output)
        else:
            weight_grad = None

        if bias is not None and not bias.stop_gradient:
            bias_grad = grad_output.sum(axis=[0, 1])
        else:
            bias_grad = None

        return input_grad, weight_grad, bias_grad


def quant_weight_linear(
    x,
    quant_weight,
    quant_dtype,
    quantization_config,
    weight_quantize_algo,
    dtype,
    quant_scale=None,
    quant_state=None,
    bias=None,
):
    return QuantizationLinearFunc.apply(
        x, quant_weight, bias, quant_scale, quant_state, quant_dtype, quantization_config, weight_quantize_algo, dtype
    )


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
        if self.weight_quantize_algo in ["weight_only_int8", "weight_only_int4", "llm.int8", "a8w8linear"]:
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
        output = quant_weight_linear(
            x=x,
            quant_weight=self.quant_weight,
            quant_dtype=self.quant_dtype,
            quantization_config=self.quantization_config,
            weight_quantize_algo=self.weight_quantize_algo,
            dtype=self._dtype,
            quant_scale=self.quant_scale,
            quant_state=(self.qquant_scale, self.double_quant_scale, self.quant_scale_offset)
            if (self.weight_quantize_algo in ["fp4", "nf4"] and self.quantization_config.qlora_weight_double_quant)
            else None,
            bias=self.bias,
        )
        return output


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
        sequence_parallel=False,
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
        self.sequence_parallel = sequence_parallel
        if self.sequence_parallel and self.gather_output:
            raise ValueError("Sequence parallel does not support gather_output")

        # PaddlePaddle dosen't support Int4 data type, one Int8 data represents two Int4 data.
        if self.weight_quantize_algo in ["weight_only_int8", "weight_only_int4", "llm.int8", "a8w8linear"]:
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
            if self.sequence_parallel:
                input_parallel = AllGatherOp.apply(x)
            else:
                input_parallel = mp_ops._c_identity(
                    x,
                    group=self.model_parallel_group,
                    skip_c_identity_dynamic=self.mp_skip_c_identity,
                )
        else:
            input_parallel = x

        output_parallel = quant_weight_linear(
            x=input_parallel,
            quant_weight=self.quant_weight,
            quant_dtype=self.quant_dtype,
            quantization_config=self.quantization_config,
            weight_quantize_algo=self.weight_quantize_algo,
            dtype=self._dtype,
            quant_scale=self.quant_scale,
            quant_state=(self.qquant_scale, self.double_quant_scale, self.quant_scale_offset)
            if (self.weight_quantize_algo in ["fp4", "nf4"] and self.quantization_config.qlora_weight_double_quant)
            else None,
            bias=self.bias,
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
        sequence_parallel=False,
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
        self.sequence_parallel = sequence_parallel
        if not self.input_is_parallel and self.sequence_parallel:
            raise ValueError("Sequence parallel only support input_is_parallel.")

        # PaddlePaddle dosen't support Int4 data type, one Int8 data represents two Int4 data.
        # paddle.nn.quant.weight_quantize will transpose in_features and out_features.
        if self.weight_quantize_algo in ["weight_only_int8", "weight_only_int4", "llm.int8", "a8w8linear"]:
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
            output_parallel = quant_weight_linear(
                x=input_parallel,
                quant_weight=self.quant_weight,
                quant_dtype=self.quant_dtype,
                quantization_config=self.quantization_config,
                weight_quantize_algo=self.weight_quantize_algo,
                dtype=self._dtype,
                quant_scale=self.quant_scale,
                quant_state=(self.qquant_scale, self.double_quant_scale, self.quant_scale_offset)
                if (self.weight_quantize_algo in ["fp4", "nf4"] and self.quantization_config.qlora_weight_double_quant)
                else None,
                bias=None,
            )
            if self.sequence_parallel:
                output_ = ReduceScatterOp.apply(output_parallel)
            else:
                output_ = mp_ops._mp_allreduce(
                    output_parallel,
                    group=self.model_parallel_group,
                    use_calc_stream=True,
                    use_model_parallel=True,
                    skip_c_identity_dynamic=self.mp_skip_c_identity,
                )
            output = output_ + self.bias if self.bias is not None else output_
        else:
            output = quant_weight_linear(
                x=input_parallel,
                quant_weight=self.quant_weight,
                quant_dtype=self.quant_dtype,
                quantization_config=self.quantization_config,
                weight_quantize_algo=self.weight_quantize_algo,
                dtype=self._dtype,
                quant_scale=self.quant_scale,
                quant_state=(self.qquant_scale, self.double_quant_scale, self.quant_scale_offset)
                if (self.weight_quantize_algo in ["fp4", "nf4"] and self.quantization_config.qlora_weight_double_quant)
                else None,
                bias=self.bias,
            )

        return output
