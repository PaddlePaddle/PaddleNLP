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
from paddle.nn.quant import weight_dequantize, weight_only_linear, weight_quantize

from ...quantization.qlora import qlora_weight_dequantize, qlora_weight_quantize
from ...quantization.quantization_linear import (
    ColumnParallelQuantizationLinear,
    QuantizationLinear,
    RowParallelQuantizationLinear,
)
from .utils import rng_ctx


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
        block_size=64,
        double_quant_block_size=256,
        double_quant=False,
        qquant_scale_attr=None,
        double_quant_scale_attr=None,
        quant_sacle_offset_attr=None,
        quant_scale_attr=None,
        llm_int8_threshold=6.0,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
    ):
        super().__init__(
            in_features,
            out_features,
            quant_algo,
            dtype,
            weight_attr,
            scale_attr,
            bias_attr,
            block_size,
            double_quant_block_size,
            double_quant,
            qquant_scale_attr,
            double_quant_scale_attr,
            quant_sacle_offset_attr,
            quant_scale_attr,
            llm_int8_threshold,
        )

        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        if self.quant_algo == "llm.int8":
            raise NotImplementedError("llm.int8 not yet support lora strategy.")
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        # Mark the weight as unmerged
        self.merged = False
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
        self.weight = None
        self.scaling = self.lora_alpha / self.r
        self.disable_lora = False

    def dequantize_weight(self):
        if self.quant_algo in ["fp4", "nf4"]:
            new_weight = (
                qlora_weight_dequantize(
                    quant_weight=self.quant_weight,
                    quant_algo=self.quant_algo,
                    state=(self.qquant_scale, self.double_quant_scale, self.quant_scale_offset)
                    if self.double_quant
                    else self.quant_scale,
                    double_quant=self.double_quant,
                    block_size=self.block_size,
                    double_quant_block_size=self.double_quant_block_size,
                )
                .cast(self._dtype)
                .reshape([self.in_features, self.out_features])
            )
        elif self.quant_algo in ["weight_only_int8"]:
            new_weight = weight_dequantize(self.quant_weight, self.quant_scale, self.quant_algo, self._dtype)
        else:
            raise NotImplementedError(f"{self.quant_algo} not yet support lora merge strategy.")
        return new_weight

    def quantize_weight(self, new_weight):
        if self.quant_algo in ["fp4", "nf4"]:
            print("self.quant_weight", self.quant_weight)
            quant_weight, quant_state = qlora_weight_quantize(
                weight=new_weight,
                quant_algo=self.quant_algo,
                double_quant=self.double_quant,
                block_size=self.block_size,
                double_quant_block_size=self.double_quant_block_size,
                return_dict=False,
            )
            print("quant_weight", quant_weight)
            self.quant_weight.set_value(quant_weight)
            if self.double_quant:
                qquant_scale, double_quant_scale, quant_sacle_offset = quant_state
                self.qquant_scale.set_value(qquant_scale)
                self.double_quant_scale.set_value(double_quant_scale)
                self.quant_sacle_offset.set_value(quant_sacle_offset)
            else:
                quant_scale = quant_state
                self.quant_scale.set_value(quant_scale)
        elif self.quant_algo in ["weight_only_int8"]:
            quant_weight, quant_scale = weight_quantize(new_weight, self.quant_algo)
            self.quant_weight.set_value(quant_weight)
            self.quant_scale.set_value(quant_scale)
        else:
            raise NotImplementedError(f"{self.quant_algo} not yet support lora merge strategy.")

    def unmerge(self):
        if self.merged:
            # Make sure that the weights are not merged
            new_weight = self.dequantize_weight()
            new_weight -= self.lora_A @ self.lora_B * self.scaling
            self.quantize_weight(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            # Merge the weights and mark it
            new_weight = self.dequantize_weight()
            new_weight += self.lora_A @ self.lora_B * self.scaling
            self.quantize_weight(new_weight)
            self.merged = True

    def forward(self, x: paddle.Tensor):
        result = super().forward(x)
        if not self.merged and not self.disable_lora:
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
        if self.quant_algo in ["fp4", "nf4"]:
            raise NotImplementedError(f"{self.quant_algo} not yet support tensor parallelism.")

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
        with rng_ctx(self.is_mp, paddle.in_dynamic_mode()):
            self.lora_B = self.create_parameter(
                shape=[r, self.output_size_per_partition],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.Constant(value=0.0),
            )
        self.lora_B.is_distributed = True
        self.lora_B.split_axis = 1
        self.scaling = self.lora_alpha / self.r
        self.disable_lora = False
        # Mark the weight as unmerged
        self.merged = False

    def forward(self, x):

        result_mp = super().forward(x)

        if not self.disable_lora or not self.merged:
            input_a = self.lora_dropout(x) @ self.lora_A
            input_a_mp = mp_ops._c_identity(input_a, group=self.model_parallel_group)
            delta_mp = (input_a_mp @ self.lora_B) * self.scaling
            result_mp += delta_mp

        if self.gather_output and self.is_mp:
            result = mp_ops._c_concat(result_mp, group=self.model_parallel_group)
        else:
            result = result_mp
        return result

    def dequantize_weight(self):
        if self.quant_algo in ["fp4", "nf4"]:
            new_weight = (
                qlora_weight_dequantize(
                    quant_weight=self.quant_weight,
                    quant_algo=self.quant_algo,
                    state=(self.qquant_scale, self.double_quant_scale, self.quant_scale_offset)
                    if self.double_quant
                    else self.quant_scale,
                    double_quant=self.double_quant,
                    block_size=self.block_size,
                    double_quant_block_size=self.double_quant_block_size,
                )
                .cast(self._dtype)
                .reshape([self.in_features, self.out_features])
            )
        elif self.quant_algo in ["weight_only_int8"]:
            new_weight = weight_dequantize(self.quant_weight, self.quant_scale, self.quant_algo, self._dtype)
        else:
            raise NotImplementedError(f"{self.quant_algo} not yet support lora merge strategy.")
        return new_weight

    def quantize_weight(self, new_weight):
        if self.quant_algo in ["fp4", "nf4"]:
            quant_weight, quant_state = qlora_weight_quantize(
                weight=new_weight,
                quant_algo=self.quant_algo,
                double_quant=self.double_quant,
                block_size=self.block_size,
                double_quant_block_size=self.double_quant_block_size,
                return_dict=False,
            )
            self.quant_weight.set_value(quant_weight)
            if self.double_quant:
                qquant_scale, double_quant_scale, quant_sacle_offset = quant_state
                self.qquant_scale.set_value(qquant_scale)
                self.double_quant_scale.set_value(double_quant_scale)
                self.quant_sacle_offset.set_value(quant_sacle_offset)
            else:
                quant_scale = quant_state
                self.quant_scale.set_value(quant_scale)
        elif self.quant_algo in ["weight_only_int8"]:
            quant_weight, quant_scale = weight_quantize(new_weight, self.quant_algo)
            self.quant_weight.set_value(quant_weight)
            self.quant_scale.set_value(quant_scale)
        else:
            raise NotImplementedError(f"{self.quant_algo} not yet support lora merge strategy.")

    def unmerge(self):
        if self.merged:
            # Make sure that the weights are not merged
            new_weight = self.dequantize_weight()
            new_weight -= self.lora_A @ self.lora_B * self.scaling
            self.quantize_weight(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            # Merge the weights and mark it
            new_weight = self.dequantize_weight()
            new_weight += self.lora_A @ self.lora_B * self.scaling
            self.quantize_weight(new_weight)
            self.merged = True


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
        if self.quant_algo == "llm.int8":
            raise NotImplementedError("llm.int8 not yet support lora strategy.")
        if self.quant_algo in ["fp4", "nf4"]:
            raise NotImplementedError(f"{self.quant_algo} not yet support tensor parallelism.")
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        # Actual trainable parameters
        with rng_ctx(self.is_mp, paddle.in_dynamic_mode()):
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
        self.disable_lora = False
        self.merged = False

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
        if not self.disable_lora or not self.merged:
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

    def dequantize_weight(self):
        if self.quant_algo in ["fp4", "nf4"]:
            new_weight = (
                qlora_weight_dequantize(
                    quant_weight=self.quant_weight,
                    quant_algo=self.quant_algo,
                    state=(self.qquant_scale, self.double_quant_scale, self.quant_scale_offset)
                    if self.double_quant
                    else self.quant_scale,
                    double_quant=self.double_quant,
                    block_size=self.block_size,
                    double_quant_block_size=self.double_quant_block_size,
                )
                .cast(self._dtype)
                .reshape([self.in_features, self.out_features])
            )
        elif self.quant_algo in ["weight_only_int8"]:
            new_weight = weight_dequantize(self.quant_weight, self.quant_scale, self.quant_algo, self._dtype)
        else:
            raise NotImplementedError(f"{self.quant_algo} not yet support lora merge strategy.")
        return new_weight

    def quantize_weight(self, new_weight):
        if self.quant_algo in ["fp4", "nf4"]:
            quant_weight, quant_state = qlora_weight_quantize(
                weight=new_weight,
                quant_algo=self.quant_algo,
                double_quant=self.double_quant,
                block_size=self.block_size,
                double_quant_block_size=self.double_quant_block_size,
                return_dict=False,
            )
            self.quant_weight.set_value(quant_weight)
            if self.double_quant:
                qquant_scale, double_quant_scale, quant_sacle_offset = quant_state
                self.qquant_scale.set_value(qquant_scale)
                self.double_quant_scale.set_value(double_quant_scale)
                self.quant_sacle_offset.set_value(quant_sacle_offset)
            else:
                quant_scale = quant_state
                self.quant_scale.set_value(quant_scale)
        elif self.quant_algo in ["weight_only_int8"]:
            quant_weight, quant_scale = weight_quantize(new_weight, self.quant_algo)
            self.quant_weight.set_value(quant_weight)
            self.quant_scale.set_value(quant_scale)
        else:
            raise NotImplementedError(f"{self.quant_algo} not yet support lora merge strategy.")

    def unmerge(self):
        if self.merged:
            # Make sure that the weights are not merged
            new_weight = self.dequantize_weight()
            new_weight -= self.lora_A @ self.lora_B * self.scaling
            self.quantize_weight(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            # Merge the weights and mark it
            new_weight = self.dequantize_weight()
            new_weight += self.lora_A @ self.lora_B * self.scaling
            self.quantize_weight(new_weight)
            self.merged = True
