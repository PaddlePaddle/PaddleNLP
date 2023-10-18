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

import math

import paddle
from paddle import nn
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.nn.quant import weight_only_linear

from ...utils.quantization import (
    ColumnParallelQuantizationLinear,
    QuantizationLinear,
    RowParallelQuantizationLinear,
)


class QuantizationLoRALinear(QuantizationLinear):
    """
    Quantization lora Linear layer.
    The code implementation refers to paddlenlp.peft.lora.lora_layers.LoRALinear.
    https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/peft/lora/lora_layers.py
    Compare to LoRALinear, this class keeps weight in INT8/INT4 with quant scale, and supports
    weight_only_linear for input tensor and origin weight(LoRA part still uses fp16/bf16).
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
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
    ):
        QuantizationLinear.__init__(
            self, in_features, out_features, quant_algo, dtype, weight_attr, scale_attr, bias_attr
        )
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        if self.quant_algo == "llm.int8":
            raise NotImplementedError("llm.int8 not yet support lora strategy.")
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        # Actual trainable parameters
        self.lora_A = self.create_parameter(
            shape=[in_features, r],
            dtype=self._dtype,
            is_bias=False,
            default_initializer=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity="leaky_relu"),
        )
        self.lora_B = self.create_parameter(
            shape=[r, out_features],
            dtype=self._dtype,
            is_bias=False,
            default_initializer=nn.initializer.Constant(value=0.0),
        )
        self.scaling = self.lora_alpha / self.r

    def forward(self, x: paddle.Tensor):
        result = super().forward(x)
        result += (self.lora_dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return result


class ColumnParallelQuantizationLoRALinear(ColumnParallelQuantizationLinear):
    """
    Quantization lora Linear layer with mp parallelized(column).
    The code implementation refers to paddlenlp.peft.lora.lora_layers.ColumnParallelLoRALinear.
    https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/peft/lora/lora_layers.py#L203
    Compare to ColumnParallelLoRALinear, this class keeps weight in INT8/INT4 with quant scale, and supports
    weight_only_linear for input tensor and origin weight(LoRA part still uses fp16/bf16).
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
        r=0,
        lora_alpha=1,
        lora_dropout=0.0,
        lora_A_weight_attr=None,
    ):
        ColumnParallelQuantizationLinear.__init__(
            self,
            in_features,
            out_features,
            quant_algo,
            dtype,
            weight_attr,
            scale_attr,
            bias_attr,
            gather_output,
            mp_group,
        )
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        if self.quant_algo == "llm.int8":
            raise NotImplementedError("llm.int8 not yet support lora strategy.")

        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        # Actual trainable parameters
        self.lora_A = self.create_parameter(
            shape=[in_features, r],
            dtype=self._dtype,
            is_bias=False,
            attr=lora_A_weight_attr,
        )
        self.lora_A.is_distributed = False
        self.lora_B = self.create_parameter(
            shape=[r, self.output_size_per_partition],
            dtype=self._dtype,
            is_bias=False,
            default_initializer=nn.initializer.Constant(value=0.0),
        )
        self.lora_B.is_distributed = True
        self.lora_B.split_axis = 1
        self.scaling = self.lora_alpha / self.r

    def forward(self, x):
        result_mp = super().forward(x)

        input_a = self.lora_dropout(x) @ self.lora_A
        input_a_mp = mp_ops._c_identity(input_a, group=self.model_parallel_group)
        delta_mp = (input_a_mp @ self.lora_B) * self.scaling
        result_mp += delta_mp

        if self.gather_output and self.is_mp:
            result = mp_ops._c_concat(result_mp, group=self.model_parallel_group)
        else:
            result = result_mp
        return result


class RowParallelQuantizationLoRALinear(RowParallelQuantizationLinear):
    """
    Quantization lora Linear layer with mp parallelized(row).
    The code implementation refers to paddlenlp.peft.lora.lora_layers.RowParallelLoRALinear.
    https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/peft/lora/lora_layers.py#L99
    Compare to RowParallelLoRALinear, this class keeps weight in INT8/INT4 with quant scale, and supports
    weight_only_linear for input tensor and origin weight(LoRA part still uses fp16/bf16).
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
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
    ):
        RowParallelQuantizationLinear.__init__(
            self,
            in_features,
            out_features,
            quant_algo,
            dtype,
            weight_attr,
            scale_attr,
            bias_attr,
            input_is_parallel,
            mp_group,
        )
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        # Actual trainable parameters
        self.lora_A = self.create_parameter(
            shape=[self.input_size_per_partition, r],
            dtype=self._dtype,
            is_bias=False,
            attr=paddle.ParamAttr(
                initializer=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity="leaky_relu")
            ),
        )
        self.lora_B = self.create_parameter(
            shape=[r, self.out_features],
            dtype=self._dtype,
            is_bias=False,
            default_initializer=nn.initializer.Constant(value=0.0),
        )
        self.lora_A.is_distributed = True
        self.lora_A.split_axis = 0
        self.lora_B.is_distributed = False
        self.scaling = self.lora_alpha / self.r

    def forward(self, x: paddle.Tensor):
        if not self.input_is_parallel:
            input_mp = mp_ops._c_split(x, group=self.model_parallel_group)
        else:
            input_mp = x

        # x @ W : [bz, in_f / ws] ===> [bz, out_f]
        with paddle.amp.auto_cast(enable=False):
            result_mp = weight_only_linear(input_mp, self.quant_weight, None, self.quant_scale, self.quant_dtype)

        output = mp_ops._mp_allreduce(
            result_mp,
            group=self.model_parallel_group,
            use_calc_stream=True,
            use_model_parallel=True,
        )

        # x @ A: [bz, in_f/ ws] ===> [bz, r]
        input_mp = self.lora_dropout(input_mp) @ self.lora_A
        # all reduce to keep Lora B's gradient on different gpu consistent
        input_dup = mp_ops._mp_allreduce(
            input_mp,
            group=self.model_parallel_group,
            use_calc_stream=True,
            use_model_parallel=True,
        )
        #  @ B: [bz, r] ===> [bz, out_f]
        delta_mp = (input_dup @ self.lora_B) * self.scaling
        output += delta_mp
        output = output + self.bias if self.bias is not None else output
        return output
