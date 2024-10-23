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
import paddle.nn as nn
import paddle.nn.functional as F


# borrow heavily from:
# https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/lokr.py
class LoKrLinear(nn.Linear):
    # LoKA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int = 0,
        lokr_alpha: float = 0.0,  # self.scale is determined by lokr_alpha/lora_dim
        factor: int = -1,
        decompose_both: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(lora_dim, int) or lora_dim <= 0:
            raise ValueError("Lora dimension lora_dim should be a positive integer")
        self.lora_dim = lora_dim
        self.use_w1 = False
        self.use_w2 = False
        # Mark the weight as unmerged
        self.merged = False
        in_m, in_n = factorization(in_features, factor)
        out_l, out_k = factorization(out_features, factor)
        shape = ((out_l, out_k), (in_m, in_n))
        self.op = F.linear

        # determining lokr_alpha
        if isinstance(lokr_alpha, paddle.Tensor):
            lokr_alpha = lokr_alpha.detach().float().numpy()  # without casting, bf16 causes error
        lokr_alpha = lora_dim if lokr_alpha is None or lokr_alpha == 0 else lokr_alpha
        if self.use_w2 and self.use_w1:
            lokr_alpha = lora_dim
        self.scale = lokr_alpha / self.lora_dim

        self.register_buffer("lokr_alpha", paddle.to_tensor(lokr_alpha))

        # Actual trainable parameters
        if decompose_both and lora_dim < max(shape[0][0], shape[1][0]) / 2:
            self.lokr_w1_a = self.create_parameter(
                shape=[shape[0][0], lora_dim],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.KaimingUniform(
                    negative_slope=math.sqrt(5), nonlinearity="leaky_relu"
                ),
            )
            self.lokr_w1_b = self.create_parameter(
                shape=[lora_dim, shape[1][0]],
                dtype=self._dtype,
                is_bias=False,
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0),
                ),
            )
        else:
            self.use_w1 = True
            self.lokr_w1 = self.create_parameter(
                shape=[shape[0][0], shape[1][0]],
                dtype=self._dtype,
                is_bias=False,
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0),
                ),
            )  # a*c, 1-mode

        if lora_dim < max(shape[0][1], shape[1][1]) / 2:
            self.lokr_w2_a = self.create_parameter(shape=[shape[0][1], lora_dim])
            self.lokr_w2_b = self.create_parameter(
                shape=[lora_dim, shape[1][1]],
                dtype=self._dtype,
                is_bias=False,
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0),
                ),
            )
            # w1 ⊗ (w2_a x w2_b) = (a, b)⊗((c, dim)x(dim, d)) = (a, b)⊗(c, d) = (ac, bd)
        else:
            self.use_w2 = True
            self.lokr_w2 = self.create_parameter(
                shape=[shape[0][1], shape[1][1]],
                dtype=self._dtype,
                is_bias=False,
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0),
                ),
            )
        adapter_weight = make_kron(
            self.lokr_w1 if self.use_w1 else self.lokr_w1_a @ self.lokr_w1_b,
            (self.lokr_w2 if self.use_w2 else self.lokr_w2_a @ self.lokr_w2_b),
            paddle.to_tensor(self.scale),
        )  # adapter param weight initialization to be zeros
        assert paddle.sum(paddle.isnan(adapter_weight)) == 0, "weight is nan"
        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        self.disable_lokr = False

    def get_adapter_weight(self, orig_weight=None):
        weight = make_kron(
            self.lokr_w1 if self.use_w1 else self.lokr_w1_a @ self.lokr_w1_b,
            (self.lokr_w2 if self.use_w2 else self.lokr_w2_a @ self.lokr_w2_b),
            paddle.to_tensor(self.scale),
        )
        if orig_weight is not None:
            weight = weight.reshape(orig_weight.shape)
        return weight

    def merge(self):
        if not self.merged:
            adapter_weight = self.get_weight()
            new_weight = self.weight + adapter_weight * self.scale  # core code
            self.weight.set_value(new_weight)
            self.merged = True

    def unmerge(self):
        if self.merged:
            adapter_weight = self.get_weight()
            new_weight = self.weight - adapter_weight * self.scale  # core code
            self.weight.set_value(new_weight)
            self.merged = False

    def forward(self, input: paddle.Tensor):  # core code
        if self.merged:
            result = self.op(x=input, weight=self.weight, bias=self.bias, name=self.name)
        else:
            result = self.op(x=input, weight=self.weight, bias=self.bias, name=self.name)
            result += self.op(x=input, weight=self.get_adapter_weight())
        return result

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


# Below code is a direct copy from https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/lokr.py#L11
def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """
    return a tuple of two value of input dimension decomposed by the number closest to factor
    second value is higher or equal than first value.

    In LoRA with Kroneckor Product, first value is a value for weight scale.
    secon value is a value for weight.

    Becuase of non-commutative property, A⊗B ≠ B⊗A. Meaning of two matrices is slightly different.

    examples)
    factor
        -1               2                4               8               16               ...
    127 -> 127, 1   127 -> 127, 1    127 -> 127, 1   127 -> 127, 1   127 -> 127, 1
    128 -> 16, 8    128 -> 64, 2     128 -> 32, 4    128 -> 16, 8    128 -> 16, 8
    250 -> 125, 2   250 -> 125, 2    250 -> 125, 2   250 -> 125, 2   250 -> 125, 2
    360 -> 45, 8    360 -> 180, 2    360 -> 90, 4    360 -> 45, 8    360 -> 45, 8
    512 -> 32, 16   512 -> 256, 2    512 -> 128, 4   512 -> 64, 8    512 -> 32, 16
    1024 -> 32, 32  1024 -> 512, 2   1024 -> 256, 4  1024 -> 128, 8  1024 -> 64, 16
    """

    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        return m, n
    if factor == -1:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def make_kron(w1, w2, scale):
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    rebuild = paddle.kron(w1, w2)

    return rebuild * scale
