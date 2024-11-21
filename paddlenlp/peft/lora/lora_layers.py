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
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)

from ...transformers import linear_utils

ColumnSequenceParallelLinear = linear_utils.ColumnSequenceParallelLinear
RowSequenceParallelLinear = linear_utils.RowSequenceParallelLinear

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        AllGatherOp,
        ReduceScatterOp,
        mark_as_sequence_parallel_parameter,
    )
except:
    AllGatherOp = None
    ReduceScatterOp = None
    mark_as_sequence_parallel_parameter = None


from ...transformers.mc2_parallel_linear import (
    MC2ColumnParallelCoreLinear,
    MC2ColumnSeqParallelCoreLinear,
    MC2RowParallelCoreLinear,
    MC2RowSeqParallelCoreLinear,
)
from .lora_quick_layers import quick_lora
from .utils import rng_ctx


class LoRALinear(nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        use_quick_lora: bool = False,
        rslora: bool = False,
        lora_plus_scale: float = 1.0,
        pissa: bool = False,
        lora_use_mixer: bool = False,
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
        self.pissa = pissa
        self.lora_use_mixer = lora_use_mixer

        # Actual trainable parameters
        self.lora_A = self.create_parameter(
            shape=[in_features, r],
            dtype=self._dtype,
            is_bias=False,
            default_initializer=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity="leaky_relu"),
        )
        if self.lora_use_mixer:
            self.lora_AB = self.create_parameter(
                shape=[r, r],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.KaimingUniform(
                    negative_slope=math.sqrt(5), nonlinearity="leaky_relu"
                ),
            )
        self.lora_B = self.create_parameter(
            shape=[r, out_features],
            dtype=self._dtype,
            is_bias=False,
            attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0),
                learning_rate=lora_plus_scale,
            ),
        )
        self.apply_pissa = False

        if not rslora and not pissa:
            self.scaling = self.lora_alpha / self.r
        elif pissa:
            self.scaling = 1.0
        else:
            self.scaling = self.lora_alpha / math.sqrt(self.r)

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        self._use_quick_lora = use_quick_lora and lora_dropout == 0.0
        self.disable_lora = False

    @property
    def use_quick_lora(self):
        return self._use_quick_lora and self.training and not self.merged

    def pissa_init(self, rank):
        weight = self.weight
        dtype = weight.dtype
        if dtype != paddle.float32:
            weight = weight.astype(paddle.float32)

        U, S, Vh = paddle.linalg.svd(weight.data, full_matrices=False)
        Ur = U[:, :rank]
        Sr = S[:rank]
        Vhr = Vh[:rank]

        lora_A = Ur @ paddle.diag(paddle.sqrt(Sr))
        lora_B = paddle.diag(paddle.sqrt(Sr)) @ Vhr
        self.lora_A.set_value(lora_A.astype(dtype))
        self.lora_B.set_value(lora_B.astype(dtype))
        res = weight.data - lora_A @ lora_B
        weight = res.astype(dtype)
        self.weight.set_value(weight)

    def merge(self):
        if not self.merged:
            if self.lora_use_mixer:
                new_weight = self.weight + self.lora_A @ self.lora_AB @ self.lora_B * self.scaling
            else:
                new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = True

    def unmerge(self):
        if self.merged:
            if self.lora_use_mixer:
                new_weight = self.weight - self.lora_A @ self.lora_AB @ self.lora_B * self.scaling
            else:
                new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def forward(self, input: paddle.Tensor, *args, **kwargs):
        if not self.apply_pissa and self.pissa:
            self.pissa_init(self.r)
            self.apply_pissa = True
        if self.disable_lora or self.merged:
            result = F.linear(x=input, weight=self.weight, bias=self.bias, name=self.name)
        elif self.use_quick_lora:
            # Use the quick lora implementation
            result = quick_lora(input, self.lora_A, self.lora_B, self.weight, self.bias, self.scaling)
        else:
            result = F.linear(x=input, weight=self.weight, bias=self.bias, name=self.name)
            if self.lora_use_mixer:
                result += (self.lora_dropout(input) @ self.lora_A @ self.lora_AB @ self.lora_B) * self.scaling
            else:
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
        rslora: bool = False,
        lora_plus_scale: float = 1.0,
        use_quick_lora: bool = False,
        pissa: bool = False,
        **kwargs
    ):
        RowParallelLinear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")

        if pissa:
            raise ValueError("Pissa is not supported in model parallel by now")

        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False

        # compatible
        self.name = self._name

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
            attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0),
                learning_rate=lora_plus_scale,
            ),
        )

        self.lora_A.is_distributed = True
        self.lora_A.split_axis = 0
        self.lora_B.is_distributed = False
        if not rslora:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = self.lora_alpha / math.sqrt(self.r)

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        self._use_quick_lora = use_quick_lora and lora_dropout == 0.0
        self.disable_lora = False

    @property
    def use_quick_lora(self):
        return self._use_quick_lora and self.training and not self.merged

    def unmerge(self):
        if self.merged:
            new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, x: paddle.Tensor):
        if not self.input_is_parallel:
            input_mp = mp_ops._c_split(x, group=self.model_parallel_group)
        else:
            input_mp = x
        if self.disable_lora or self.merged:
            # x @ W : [bz, in_f / ws] ===> [bz, out_f]
            if MC2RowParallelCoreLinear is None:
                result_mp = F.linear(x=input_mp, weight=self.weight, name=self.name)
                output = mp_ops._mp_allreduce(
                    result_mp,
                    group=self.model_parallel_group,
                    use_calc_stream=True,
                    use_model_parallel=True,
                )
            else:
                output = MC2RowParallelCoreLinear.apply(input_mp, self.weight, self.model_parallel_group)
            output = output + self.bias if self.bias is not None else output
        elif self.use_quick_lora:
            # Use the quick lora implementation
            result_mp = quick_lora(
                input_mp,
                self.lora_A,
                self.lora_B,
                self.weight,
                self.bias,
                self.scaling,
                is_row=True,
                group=self.model_parallel_group,
                world_size=self.world_size,
            )
            output = mp_ops._mp_allreduce(
                result_mp,
                group=self.model_parallel_group,
                use_calc_stream=True,
                use_model_parallel=True,
            )
        else:
            # x @ W : [bz, in_f / ws] ===> [bz, out_f]
            if MC2RowParallelCoreLinear is None:
                result_mp = F.linear(x=input_mp, weight=self.weight, name=self.name)
                output = mp_ops._mp_allreduce(
                    result_mp,
                    group=self.model_parallel_group,
                    use_calc_stream=True,
                    use_model_parallel=True,
                )
            else:
                output = MC2RowParallelCoreLinear.apply(input_mp, self.weight, self.model_parallel_group)

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

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


class RowSequenceParallelLoRALinear(RowSequenceParallelLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        rslora: bool = False,
        lora_plus_scale: float = 1.0,
        use_quick_lora: bool = False,
        **kwargs
    ):
        RowSequenceParallelLinear.__init__(self, in_features, out_features, **kwargs)
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

        # compatible
        self.name = self._name

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
            attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0),
                learning_rate=lora_plus_scale,
            ),
        )

        self.lora_A.is_distributed = True
        self.lora_A.split_axis = 0
        self.lora_B.is_distributed = False
        mark_as_sequence_parallel_parameter(self.lora_B)
        if not rslora:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = self.lora_alpha / math.sqrt(self.r)

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        self._use_quick_lora = use_quick_lora and lora_dropout == 0.0
        self.disable_lora = False

    @property
    def use_quick_lora(self):
        # TODO(@gexiao): support qlora
        return False  # self._use_quick_lora and self.training and not self.merged

    def unmerge(self):
        if self.merged:
            new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, x: paddle.Tensor):
        if not self.input_is_parallel:
            input_mp = mp_ops._c_split(x, group=self.model_parallel_group)
        else:
            input_mp = x

        if MC2RowSeqParallelCoreLinear is None:
            output_parallel = self.linear(input_mp, self.weight, name=self._name)
            output_ = ReduceScatterOp.apply(output_parallel)
            result_mp = output_ + self.bias if self.bias is not None else output_
        else:
            output_ = MC2RowSeqParallelCoreLinear.apply(input_mp, self.weight, self.model_parallel_group)
            result_mp = output_ + self.bias if self.bias is not None else output_

        if not self.merged and not self.disable_lora:
            input_mp = self.lora_dropout(input_mp)
            # TODO(@gexiao): temporary workaround for deterministic calculation
            if True or MC2RowSeqParallelCoreLinear is None:
                input_mp = input_mp @ self.lora_A
                input_mp = ReduceScatterOp.apply(input_mp)
            else:
                input_mp = MC2RowSeqParallelCoreLinear.apply(input_mp, self.lora_A, self.model_parallel_group)
            delta_mp = (input_mp @ self.lora_B) * self.scaling
            result_mp += delta_mp
        return result_mp

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
        rslora: bool = False,
        lora_plus_scale: float = 1.0,
        lora_A_weight_attr: Optional[paddle.ParamAttr] = None,
        use_quick_lora: bool = False,
        pissa: bool = False,
        **kwargs
    ):
        ColumnParallelLinear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")

        if pissa:
            raise ValueError("Pissa is not supported in model parallel by now")

        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False

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
        with rng_ctx(self.is_mp, paddle.in_dynamic_mode()):
            self.lora_B = self.create_parameter(
                shape=[r, self.output_size_per_partition],
                dtype=self._dtype,
                is_bias=False,
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0),
                    learning_rate=lora_plus_scale,
                ),
            )

        self.lora_B.is_distributed = True
        self.lora_B.split_axis = 1
        if not rslora:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = self.lora_alpha / math.sqrt(self.r)

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        self._use_quick_lora = use_quick_lora and lora_dropout == 0.0
        self.disable_lora = False

    @property
    def use_quick_lora(self):
        return self._use_quick_lora and self.training and not self.merged

    def unmerge(self):
        if self.merged:
            # Make sure that the weights are not merged
            new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            # Merge the weights and mark it
            new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, input: paddle.Tensor):
        if self.disable_lora or self.merged:
            if MC2ColumnParallelCoreLinear is None:
                input_mp = mp_ops._c_identity(input, group=self.model_parallel_group)
                result_mp = F.linear(x=input_mp, weight=self.weight, bias=self.bias, name=self.name)
            else:
                res_mp = MC2ColumnParallelCoreLinear.apply(input, self.weight, self.model_parallel_group)
                result_mp = (res_mp + self.bias) if self.bias is not None else res_mp

        elif self.use_quick_lora:
            # Use the quick lora implementation
            input_mp = mp_ops._c_identity(input, group=self.model_parallel_group) if self.is_mp else input
            result_mp = quick_lora(
                input_mp,
                self.lora_A,
                self.lora_B,
                self.weight,
                self.bias,
                self.scaling,
                is_column=True,
                group=self.model_parallel_group,
                world_size=self.world_size,
            )
        else:
            if MC2ColumnParallelCoreLinear is None:
                input_mp = mp_ops._c_identity(input, group=self.model_parallel_group)
                result_mp = F.linear(x=input_mp, weight=self.weight, bias=self.bias, name=self.name)
            else:
                res_mp = MC2ColumnParallelCoreLinear.apply(input, self.weight, self.model_parallel_group)
                result_mp = (res_mp + self.bias) if self.bias is not None else res_mp

            input_a = self.lora_dropout(input) @ self.lora_A
            if MC2ColumnParallelCoreLinear is None:
                input_a_mp = mp_ops._c_identity(input_a, group=self.model_parallel_group)
                delta_mp = (input_a_mp @ self.lora_B) * self.scaling
            else:
                tmp = MC2ColumnParallelCoreLinear.apply(input_a, self.lora_B, self.model_parallel_group)
                delta_mp = tmp * self.scaling
            result_mp += delta_mp

        if self.gather_output and self.is_mp:
            result = mp_ops._c_concat(result_mp, group=self.model_parallel_group)
        else:
            result = result_mp
        return result

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


class ColumnSequenceParallelLoRALinear(ColumnSequenceParallelLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        rslora: bool = False,
        lora_plus_scale: float = 1.0,
        lora_A_weight_attr: Optional[paddle.ParamAttr] = None,
        use_quick_lora: bool = False,
        **kwargs
    ):
        ColumnSequenceParallelLinear.__init__(self, in_features, out_features, **kwargs)
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
        mark_as_sequence_parallel_parameter(self.lora_A)

        with rng_ctx(self.is_mp, paddle.in_dynamic_mode()):
            self.lora_B = self.create_parameter(
                shape=[r, self.output_size_per_partition],
                dtype=self._dtype,
                is_bias=False,
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0),
                    learning_rate=lora_plus_scale,
                ),
            )

        self.lora_B.is_distributed = True
        self.lora_B.split_axis = 1
        if not rslora:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = self.lora_alpha / math.sqrt(self.r)

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        self._use_quick_lora = use_quick_lora and lora_dropout == 0.0
        self.disable_lora = False

    @property
    def use_quick_lora(self):
        # TODO(@gexiao): support qlora
        return False  # self._use_quick_lora and self.training and not self.merged

    def unmerge(self):
        if self.merged:
            new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, x: paddle.Tensor):
        if MC2ColumnSeqParallelCoreLinear is None:
            if self.is_mp:
                input_parallel = AllGatherOp.apply(x)
            else:
                input_parallel = x
            result_mp = self.linear(input_parallel, self.weight, self.bias, name=self._name)
        else:
            result_mp = MC2ColumnSeqParallelCoreLinear.apply(x, self.weight, self.model_parallel_group)
            if self.bias is not None:
                result_mp += self.bias

        if not self.merged and not self.disable_lora:
            input_a = self.lora_dropout(x) @ self.lora_A
            # TODO(@gexiao): temporary workaround for deterministic calculation
            if True or MC2ColumnSeqParallelCoreLinear is None:
                input_a = AllGatherOp.apply(input_a)
                delta_mp = (input_a @ self.lora_B) * self.scaling
            else:
                input_a = MC2ColumnSeqParallelCoreLinear.apply(input_a, self.lora_B, self.model_parallel_group)
                delta_mp = input_a * self.scaling
            result_mp += delta_mp

        if self.gather_output and self.is_mp:
            result = mp_ops._c_concat(result_mp, group=self.model_parallel_group)
        else:
            result = result_mp
        return result

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


class LoRAConv2D(nn.Conv2D):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs
    ):
        nn.Conv2D.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
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

        # Actual trainable parameters
        lora_A = nn.Conv2D(
            in_channels,
            r,
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=self._padding,
            weight_attr=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity="leaky_relu"),
            bias_attr=False,
        )
        self.lora_A = lora_A.weight
        self.lora_A_forward = lambda x: nn.Conv2D.__call__(lora_A, x)
        lora_B = nn.Conv2D(
            r,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            weight_attr=nn.initializer.Constant(value=0.0),
            bias_attr=False,
        )
        self.lora_B_forward = lambda x: nn.Conv2D.__call__(lora_B, x)
        self.lora_B = lora_B.weight
        self.scaling = lora_alpha / r

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        if self.bias is not None:
            self.bias.stop_gradient = True
        self.disable_lora = False

    def unmerge(self):
        if self.merged:
            weight_A = self.lora_A.cast(dtype=self.weight.dtype)
            weight_B = self.lora_B.cast(dtype=self.weight.dtype)
            if self.weight.shape[2:4] == [1, 1]:
                # conv2d 1x1
                delta_weight = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(
                    2
                ).unsqueeze(3) * self.scaling
            else:
                # conv2d 3x3
                delta_weight = (
                    F.conv2d(
                        weight_A.transpose([1, 0, 2, 3]),
                        weight_B,
                    ).transpose([1, 0, 2, 3])
                    * self.scaling
                )
            # Make sure that the weights are not merged
            new_weight = self.weight - delta_weight
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            weight_A = self.lora_A.cast(dtype=self.weight.dtype)
            weight_B = self.lora_B.cast(dtype=self.weight.dtype)
            if self.weight.shape[2:4] == [1, 1]:
                # conv2d 1x1
                delta_weight = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(
                    2
                ).unsqueeze(3) * self.scaling
            else:
                # conv2d 3x3
                delta_weight = (
                    F.conv2d(
                        weight_A.transpose([1, 0, 2, 3]),
                        weight_B,
                    ).transpose([1, 0, 2, 3])
                    * self.scaling
                )
            # Merge the weights and mark it
            new_weight = self.weight + delta_weight
            self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, input: paddle.Tensor, *args, **kwargs):
        previous_dtype = input.dtype
        result = super().forward(input)
        if not self.merged and not self.disable_lora:
            result += (
                self.lora_B_forward(self.lora_A_forward(self.lora_dropout(input.cast(dtype=self.lora_A.dtype))))
                * self.scaling
            )
        result = result.cast(dtype=previous_dtype)
        return result

    def extra_repr(self):
        main_str = "{_in_channels}, {_out_channels}, kernel_size={_kernel_size}"
        if self._stride != [1] * len(self._stride):
            main_str += ", stride={_stride}"
        if self._padding != 0:
            main_str += ", padding={_padding}"
        if self._padding_mode != "zeros":
            main_str += ", padding_mode={_padding_mode}"
        if self.output_padding != 0:
            main_str += ", output_padding={output_padding}"
        if self._dilation != [1] * len(self._dilation):
            main_str += ", dilation={_dilation}"
        if self._groups != 1:
            main_str += ", groups={_groups}"
        main_str += ", data_format={_data_format}, rank={r}, alpha={lora_alpha}"
        return main_str.format(**self.__dict__)
