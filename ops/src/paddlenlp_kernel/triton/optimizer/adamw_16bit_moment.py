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

import triton
import triton.language as tl


@triton.jit
def adamw_kernel(
    param_ptr,
    grad_ptr,
    moment1_ptr,
    moment2_ptr,
    lr_ptr,
    beta1,
    beta2,
    epsilon,
    coeff,
    beta1_pow_ptr,
    beta2_pow_ptr,
    master_weight_ptr,
    dtype,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    if master_weight_ptr is not None:
        param = tl.load(master_weight_ptr + offsets, mask=mask)
    else:
        param = tl.load(param_ptr + offsets, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
    moment1 = tl.load(moment1_ptr + offsets, mask=mask).to(tl.float32)
    moment2 = tl.load(moment2_ptr + offsets, mask=mask).to(tl.float32)
    lr = tl.load(lr_ptr)
    beta1_pow = tl.load(beta1_pow_ptr)
    beta2_pow = tl.load(beta2_pow_ptr)

    # Weight Decay
    param *= 1.0 - lr * coeff

    # AdamW
    moment1 = beta1 * moment1 + (1.0 - beta1) * grad
    moment2 = beta2 * moment2 + (1.0 - beta2) * grad * grad
    denom = tl.sqrt(moment2) / tl.sqrt(1.0 - beta2_pow) + epsilon
    param += (moment1 / denom) * (-lr / (1 - beta1_pow))
    if dtype == 0:
        target_dtype = tl.float16
    elif dtype == 1:
        target_dtype = tl.bfloat16
    else:
        target_dtype = tl.float32
    target_dtype = tl.bfloat16

    # Update param
    if master_weight_ptr is not None:
        tl.store(master_weight_ptr + offsets, param, mask=mask)
        tl.store(param_ptr + offsets, param.to(target_dtype), mask=mask)
    else:
        tl.store(param_ptr + offsets, param.to(target_dtype), mask=mask)
    tl.store(moment1_ptr + offsets, moment1.to(target_dtype), mask=mask)
    tl.store(moment2_ptr + offsets, moment2.to(target_dtype), mask=mask)


def adamw_16bit_moment(
    param,
    grad,
    learning_rate,
    moment1,
    moment2,
    beta1_pow,
    beta2_pow,
    master_weight,
    skip_update,
    beta1,
    beta2,
    epsilon,
    lr_ratio,
    coeff,
    with_decay,
    multi_precision,
):
    if skip_update:
        return
    if not with_decay:
        coeff = 0.0
    if not multi_precision:
        master_weight = None
    lr = learning_rate * lr_ratio

    N = param.numel().item()
    BLOCK_SIZE = 512
    grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)
    if str(param.dtype) == "paddle.float16":
        dtype = 0
    elif str(param.dtype) == "paddle.bfloat16":
        dtype = 1
    else:
        dtype = 2
    adamw_kernel[grid](
        param,
        grad,
        moment1,
        moment2,
        lr,
        beta1,
        beta2,
        epsilon,
        coeff,
        beta1_pow,
        beta2_pow,
        master_weight,
        dtype,
        N,
        BLOCK_SIZE,
    )
    beta1_pow[:], beta2_pow[:] = beta1 * beta1_pow[:], beta2 * beta2_pow[:]
