# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.nn.quant import weight_dequantize, weight_quantize

from paddlenlp.quantization.qlora import qlora_weight_quantize_dequantize

from .lora_layers import LoRALinear
from .lora_model import LoRAModel


def weight_quantize_dequantize(W: paddle.Tensor, quant_algo: str) -> paddle.Tensor:
    if W.dtype not in [paddle.float16, paddle.bfloat16]:
        old_type = W.dtype
        W = paddle.cast(W, paddle.float16)
    else:
        old_type = None
    qW, qscale = weight_quantize(W, algo=quant_algo)
    W_ = weight_dequantize(qW, qscale, algo=quant_algo)
    if old_type is not None:
        W_ = paddle.cast(W_, old_type)
    return W_


def transform_lora_layers(model: LoRAModel, quantization_config_dict: dict, sensitivity_dict: dict = None) -> None:
    for name, submodule in model.named_sublayers():
        print("transform " + name)
        if isinstance(submodule, LoRALinear):
            if sensitivity_dict is None:
                sensitivity = None
            else:
                sensitivity = sensitivity_dict.get(name[6:], None)

            num_ranks = submodule.r
            W = submodule.weight
            quant_algo = quantization_config_dict.get(name[6:], None)

            if W.dtype in [paddle.float16]:
                old_dtype = W.dtype
                W = paddle.cast(W, dtype=paddle.float32)
            else:
                old_dtype = None

            if sensitivity is not None:
                sensitivity = paddle.cast(sensitivity, dtype=W.dtype)

            Q, lora_A, lora_B = lowrand_quantized_sparse_decomposition(
                W, num_ranks, quant_algo=quant_algo, sensitivity=sensitivity
            )

            if old_dtype is not None:
                lora_A = paddle.cast(lora_A, dtype=old_dtype)
                lora_B = paddle.cast(lora_B, dtype=old_dtype)
                Q = paddle.cast(Q, dtype=old_dtype)

            scale_sqrt = math.sqrt(submodule.scaling)
            submodule.lora_A.set_value(lora_A / scale_sqrt)
            submodule.lora_B.set_value(lora_B / scale_sqrt)
            submodule.weight.set_value(Q)


def lowrand_quantized_sparse_decomposition(
    W: paddle.Tensor,
    num_ranks: int,
    num_iterations: int = 100,
    quant_algo: str = "nf4",
    sensitivity: paddle.Tensor = None,
    double_quant: bool = True,
):
    Q = paddle.zeros_like(W)
    last_error = paddle.to_tensor(float("inf"), dtype=W.dtype)
    for i in range(num_iterations):
        A = W - Q
        if sensitivity is None:
            lora_A, lora_B = svd_decomposition(A, num_ranks)
        else:
            lora_A, lora_B = weighted_svd_decomposition(A, sensitivity, num_ranks)

        if quant_algo in ["weight_only_int8"]:
            Q = weight_quantize_dequantize(W - lora_A @ lora_B, quant_algo=quant_algo)
        elif quant_algo in ["fp4", "nf4"]:
            Q = qlora_weight_quantize_dequantize(W - lora_A @ lora_B, quant_algo=quant_algo, double_quant=double_quant)
        else:
            raise NotImplementedError(f"{quant_algo} is not support.")

        W_ = Q + lora_A @ lora_B

        if sensitivity is None:
            error = paddle.linalg.norm(W - W_, p="fro")
        else:
            error = paddle.linalg.norm(paddle.sqrt(sensitivity) * (W - W_), p="fro")

        if error > last_error:
            break
        last_error = error
    return Q, lora_A, lora_B


def svd_decomposition(
    A: paddle.Tensor,
    num_ranks: int,
):
    if A.ndim != 2:
        raise ValueError(f"Expected 2D Matrix, but got {A.ndim}.")

    U, S, Vh = paddle.linalg.svd(A, full_matrices=False)
    Ur = U[:, :num_ranks]
    Sr = S[:num_ranks]
    Vhr = Vh[:num_ranks]

    lora_A = Ur @ paddle.diag(paddle.sqrt(Sr))
    lora_B = paddle.diag(paddle.sqrt(Sr)) @ Vhr
    return lora_A, lora_B


def weighted_svd_decomposition(
    A: paddle.Tensor,
    sensitivity: paddle.Tensor,
    num_ranks: int,
    normalize: bool = False,
    reduce_before_sqrt: bool = True,
):
    if A.ndim != 2 or sensitivity.ndim != 2:
        raise ValueError(f"Expected 2D Matrix, but got {A.ndim} and {sensitivity.ndim}.")
    if A.shape != sensitivity.shape:
        raise ValueError(f"Expected A.shape == W.shape, but got {A.shape} and {sensitivity.shape}.")

    if normalize is True:
        sensitivity = sensitivity / paddle.linalg.norm(sensitivity, p="fro")

    if reduce_before_sqrt is True:
        # (A.shape[0], 1)
        W1 = paddle.sqrt(paddle.mean(sensitivity, axis=1, keepdim=True))
        # (1, A.shape[1])
        W2 = paddle.sqrt(paddle.mean(sensitivity, axis=0, keepdim=True))
    else:
        # (A.shape[0], 1)
        W1 = paddle.mean(paddle.sqrt(sensitivity), axis=1, keepdim=True)
        # (1, A.shape[1])
        W2 = paddle.mean(paddle.sqrt(sensitivity), axis=0, keepdim=True)

    A_tilde = W1 * A * W2

    L1_tilde, L2_tilde = svd_decomposition(A_tilde, num_ranks=num_ranks)

    L1 = L1_tilde / W1
    L2 = L2_tilde / W2

    return L1, L2
