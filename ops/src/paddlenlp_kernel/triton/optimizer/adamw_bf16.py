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
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    param = tl.load(param_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    moment1 = tl.load(moment1_ptr + offsets, mask=mask).to(tl.float32)
    moment2 = tl.load(moment2_ptr + offsets, mask=mask).to(tl.float32)
    lr = tl.load(lr_ptr)
    beta1_pow = tl.load(beta1_pow_ptr)
    beta2_pow = tl.load(beta2_pow_ptr)

    if master_weight_ptr:
        master_weight = tl.load(master_weight_ptr + offsets, mask=mask)
        param = master_weight
    else:
        param = param.to(tl.float32)

    # Weight Decay
    param *= 1.0 - lr * coeff

    # AdamW
    moment1 = beta1 * moment1 + (1 - beta1) * grad
    moment2 = beta2 * moment2 + (1 - beta2) * grad * grad

    denom = tl.sqrt(moment2 / (1 - beta2_pow)) + epsilon

    update = (moment1 / denom) * (-lr / (1 - beta1_pow))
    param += update

    # Update param
    tl.store(moment1_ptr + offsets, moment1, mask=mask)
    tl.store(moment2_ptr + offsets, moment2, mask=mask)
    tl.store(beta1_pow_ptr + offsets, beta1 * beta1_pow, mask=mask)
    tl.store(beta2_pow_ptr + offsets, beta2 * beta2_pow, mask=mask)
    if master_weight_ptr:
        tl.store(master_weight_ptr + offsets, param, mask=mask)
        tl.store(param_ptr + offsets, param.to(tl.bfloat16), mask=mask)
    else:
        tl.store(param_ptr + offsets, param.to(tl.bfloat16), mask=mask)


def adamw_bf16(
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

    N = param.numel()
    BLOCK_SIZE = 512
    grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)

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
        N,
        BLOCK_SIZE,
    )
