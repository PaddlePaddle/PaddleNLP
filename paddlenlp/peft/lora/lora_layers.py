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
from typing import List, Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)


class LoRALinear(nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

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

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True

    def train(self):
        super().train()
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def eval(self):
        super().eval()
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, input: paddle.Tensor):
        result = F.linear(x=input, weight=self.weight, bias=self.bias, name=self.name)
        if not self.merged:
            result += (self.lora_dropout(input) @ self.lora_A @ self.lora_B) * self.scaling
        return result

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


class RowParallelLoRALinear(RowParallelLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs
    ):
        RowParallelLinear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

        # compatible
        self.name = self._name

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

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True

    def train(self):
        super().train()
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def eval(self):
        super().eval()
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, x: paddle.Tensor):
        if not self.input_is_parallel:
            input_mp = mp_ops._c_split(x, group=self.model_parallel_group)
        else:
            input_mp = x

        # x @ W : [bz, in_f / ws] ===> [bz, out_f]
        result_mp = F.linear(x=input_mp, weight=self.weight, bias=self.bias, name=self.name)

        output = mp_ops._mp_allreduce(
            result_mp,
            group=self.model_parallel_group,
            use_calc_stream=True,
            use_model_parallel=True,
        )

        if not self.merged:
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

        return output

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


class ColumnParallelLoRALinear(ColumnParallelLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        lora_A_weight_attr: Optional[paddle.ParamAttr] = None,
        **kwargs
    ):
        ColumnParallelLinear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

        # compatible
        self.name = self._name

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

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True

    def train(self):
        super().train()
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def eval(self):
        super().eval()
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, input: paddle.Tensor):
        input_mp = mp_ops._c_identity(input, group=self.model_parallel_group)
        result_mp = F.linear(x=input_mp, weight=self.weight, bias=self.bias, name=self.name)

        if not self.merged:
            input_a = self.lora_dropout(input) @ self.lora_A
            input_a_mp = mp_ops._c_identity(input_a, group=self.model_parallel_group)
            delta_mp = (input_a_mp @ self.lora_B) * self.scaling
            result_mp += delta_mp

        if self.gather_output and self.is_mp:
            result = mp_ops._c_concat(result_mp, group=self.model_parallel_group)
        else:
            result = result_mp
        return result

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


class LoRAMergedLinear(nn.Linear):
    # LoRA implemented in a dense layer  with merged linear weights for q, k, v
    def __init__(
        self,
        in_features: int,
        out_features: int,
        head_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        enable_lora: List[bool] = [False],
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        assert (
            out_features % len(enable_lora) == 0
        ), f"The length of enable_lora must divide out_features: {out_features} % {len(enable_lora)} != 0"
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        self.r = r
        self.lora_alpha = lora_alpha
        if isinstance(enable_lora, List) and all(isinstance(item, bool) for item in enable_lora):
            self.enable_lora = enable_lora
        else:
            raise TypeError("enable_lora must be a list of bools")

        self.out_features = out_features
        self.in_features = in_features
        self.head_dim = head_dim
        self.head_num = self.out_features // len(enable_lora) // self.head_dim

        # Optional dropout
        if lora_dropout > 0.0 and any:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

        # Actual trainable parameters
        if any(enable_lora):
            self.lora_A = self.create_parameter(
                shape=[in_features, r * sum(enable_lora)],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.KaimingUniform(
                    negative_slope=math.sqrt(5), nonlinearity="leaky_relu"
                ),
            )
            # Make sure lora_B is split in column for ColumnParallelLoRAMergedLinear.
            self.lora_B = self.create_parameter(
                shape=[r, out_features // len(enable_lora) * sum(enable_lora)],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.Constant(value=0.0),
            )
            self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            self.weight.stop_gradient = True

    def zero_pad_and_reshape(self, x):
        # if enable_lora is all true, then there is no need to zero pad
        if all(self.enable_lora):
            output = x
        else:
            split_output = paddle.split(x, sum(self.enable_lora), axis=-1)
            for index in range(len(self.enable_lora)):
                if self.enable_lora[index] is False:
                    split_output.insert(index, paddle.zeros_like(split_output[0]))
            output = paddle.concat(split_output, axis=-1)
        if output.dim() == 2:
            rank, out_features = output.shape
            reshape_output = (
                output.reshape([rank, len(self.enable_lora), self.head_num, self.head_dim])
                .transpose([0, 2, 1, 3])
                .reshape([rank, out_features])
            )
        else:
            batch, seq_len, out_features = output.shape
            reshape_output = (
                output.reshape([batch, seq_len, len(self.enable_lora), self.head_num, self.head_dim])
                .transpose([0, 1, 3, 2, 4])
                .reshape([batch, seq_len, out_features])
            )

        return reshape_output

    def train(self):
        super().train()
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if any(self.enable_lora):
                reshape_lora_B = (
                    self.lora_B.reshape([self.r, self.head_num, sum(self.enable_lora), self.head_dim])
                    .transpose([0, 2, 1, 3])
                    .reshape(self.lora_B.shape)
                )
                delta_weight = (
                    F.conv1d(
                        self.lora_A.T.unsqueeze(0),
                        reshape_lora_B.T.unsqueeze(-1),
                        groups=sum(self.enable_lora),
                    )
                    .squeeze(0)
                    .T
                )
                new_weight = self.weight - self.zero_pad_and_reshape(delta_weight * self.scaling)
                self.weight.set_value(new_weight)
            self.merged = False

    def eval(self):
        super().eval()
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if any(self.enable_lora):
                reshape_lora_B = (
                    self.lora_B.reshape([self.r, self.head_num, sum(self.enable_lora), self.head_dim])
                    .transpose([0, 2, 1, 3])
                    .reshape(self.lora_B.shape)
                )
                delta_weight = (
                    F.conv1d(
                        self.lora_A.T.unsqueeze(0),
                        reshape_lora_B.T.unsqueeze(-1),
                        groups=sum(self.enable_lora),
                    )
                    .squeeze(0)
                    .T
                )
                new_weight = self.weight + self.zero_pad_and_reshape(delta_weight * self.scaling)
                self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, input: paddle.Tensor):
        result = F.linear(x=input, weight=self.weight, bias=self.bias, name=self.name)
        if any(self.enable_lora) and not self.merged:
            input_a = self.lora_dropout(input) @ self.lora_A
            if input_a.dim() == 3:
                reshape_lora_B = (
                    self.lora_B.reshape([self.r, self.head_num, sum(self.enable_lora), self.head_dim])
                    .transpose([0, 2, 1, 3])
                    .reshape(self.lora_B.shape)
                )
                delta = (
                    F.conv1d(
                        input_a.transpose([0, 2, 1]),
                        reshape_lora_B.T.unsqueeze(-1),
                        groups=sum(self.enable_lora),
                    )
                ).transpose([0, 2, 1])
            else:
                raise NotImplementedError("LoRAMergedLinear only support 3D input features")

            result += self.zero_pad_and_reshape(delta * self.scaling)
        return result

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


class ColumnParallelLoRAMergedLinear(ColumnParallelLinear):
    # LoRA implemented in a dense layer  with merged linear weights for q, k, v
    def __init__(
        self,
        in_features: int,
        out_features: int,
        head_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        enable_lora: List[bool] = [False],
        lora_A_weight_attr: Optional[paddle.ParamAttr] = None,
        **kwargs
    ):
        ColumnParallelLinear.__init__(self, in_features, out_features, **kwargs)
        assert (
            self.output_size_per_partition % len(enable_lora) == 0
        ), f"The length of enable_lora must divide out_features: {self.output_size_per_partition} % {len(enable_lora)} != 0"
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        self.r = r
        self.lora_alpha = lora_alpha
        if isinstance(enable_lora, List) and all(isinstance(item, bool) for item in enable_lora):
            self.enable_lora = enable_lora
        else:
            raise TypeError("enable_lora must be a list of bools")

        self.out_features = out_features
        self.in_features = in_features
        self.head_dim = head_dim
        self.head_num = self.output_size_per_partition // len(enable_lora) // self.head_dim

        # Optional dropout
        if lora_dropout > 0.0 and any:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

        # compatible
        self.name = self._name

        # Actual trainable parameters
        if any(enable_lora):
            self.lora_A = self.create_parameter(
                shape=[in_features, r * sum(enable_lora)],
                dtype=self._dtype,
                is_bias=False,
                attr=lora_A_weight_attr,
            )
            self.lora_A.is_distributed = False
            # Make sure lora_B is split in column the same as ColumnParallelLoRALinear.
            self.lora_B = self.create_parameter(
                shape=[r, self.output_size_per_partition // len(enable_lora) * sum(enable_lora)],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.Constant(value=0.0),
            )
            self.lora_B.is_distributed = True
            self.lora_B.split_axis = 1
            self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            self.weight.stop_gradient = True

    def zero_pad_and_reshape(self, x):
        # if enable_lora is all true, then there is no need to zero pad
        if all(self.enable_lora):
            output = x
        else:
            split_output = paddle.split(x, sum(self.enable_lora), axis=-1)
            for index in range(len(self.enable_lora)):
                if self.enable_lora[index] is False:
                    split_output.insert(index, paddle.zeros_like(split_output[0]))
            output = paddle.concat(split_output, axis=-1)
        if output.dim() == 2:
            rank, out_features = output.shape
            reshape_output = (
                output.reshape([rank, len(self.enable_lora), self.head_num, self.head_dim])
                .transpose([0, 2, 1, 3])
                .reshape([rank, out_features])
            )
        else:
            batch, seq_len, out_features = output.shape
            reshape_output = (
                output.reshape([batch, seq_len, len(self.enable_lora), self.head_num, self.head_dim])
                .transpose([0, 1, 3, 2, 4])
                .reshape([batch, seq_len, out_features])
            )

        return reshape_output

    def train(self):
        super().train()
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if any(self.enable_lora):
                reshape_lora_B = (
                    self.lora_B.reshape([self.r, self.head_num, sum(self.enable_lora), self.head_dim])
                    .transpose([0, 2, 1, 3])
                    .reshape(self.lora_B.shape)
                )
                delta_weight = (
                    F.conv1d(
                        self.lora_A.T.unsqueeze(0),
                        reshape_lora_B.T.unsqueeze(-1),
                        groups=sum(self.enable_lora),
                    )
                    .squeeze(0)
                    .T
                )
                new_weight = self.weight - self.zero_pad_and_reshape(delta_weight * self.scaling)
                self.weight.set_value(new_weight)
            self.merged = False

    def eval(self):
        super().eval()
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if any(self.enable_lora):
                reshape_lora_B = (
                    self.lora_B.reshape([self.r, self.head_num, sum(self.enable_lora), self.head_dim])
                    .transpose([0, 2, 1, 3])
                    .reshape(self.lora_B.shape)
                )
                delta_weight = (
                    F.conv1d(
                        self.lora_A.T.unsqueeze(0),
                        reshape_lora_B.T.unsqueeze(-1),
                        groups=sum(self.enable_lora),
                    )
                    .squeeze(0)
                    .T
                )
                new_weight = self.weight + self.zero_pad_and_reshape(delta_weight * self.scaling)
                self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, input: paddle.Tensor):
        # [batch_size, *, in_features]
        input_mp = mp_ops._c_identity(input, group=self.model_parallel_group)
        # [batch_size, *, out_features_per_partition]
        result_mp = F.linear(x=input_mp, weight=self.weight, bias=self.bias, name=self.name)
        if any(self.enable_lora) and not self.merged:
            input_a = self.lora_dropout(input) @ self.lora_A
            input_a_mp = mp_ops._c_identity(input_a, group=self.model_parallel_group)
            if input_a.dim() == 3:
                reshape_lora_B = (
                    self.lora_B.reshape([self.r, self.head_num, sum(self.enable_lora), self.head_dim])
                    .transpose([0, 2, 1, 3])
                    .reshape(self.lora_B.shape)
                )
                delta_mp = (
                    F.conv1d(
                        input_a_mp.transpose([0, 2, 1]),
                        reshape_lora_B.T.unsqueeze(-1),
                        groups=sum(self.enable_lora),
                    )
                ).transpose([0, 2, 1])
            else:
                raise NotImplementedError("LoRAMergedLinear only support 3D input features")
            # [batch_size, *, out_features_per_partition]
            result_mp += self.zero_pad_and_reshape(delta_mp * self.scaling)

        if self.gather_output and self.is_mp:
            result_mp_list = paddle.split(result_mp, len(self.enable_lora), axis=-1)
            result_list = []
            for result_mp in result_mp_list:
                result_list.append(mp_ops._c_concat(result_mp, group=self.model_parallel_group))
            # [batch_size, *, out_features]
            result = paddle.concat(result_list, axis=-1)
        else:
            result = result_mp

        return result

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"
