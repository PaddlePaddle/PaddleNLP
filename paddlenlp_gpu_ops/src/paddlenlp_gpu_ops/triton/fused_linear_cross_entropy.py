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
"""
this code is modified from https://github.com/JonasGeiping/linear_cross_entropy_loss/blob/main/linear_cross_entropy/ops.py
"""
import paddle
import triton

from ..utils import custom_bwd, custom_fwd, get_autocast_gpu_dtype, is_autocast_enabled
from .cross_entropy import cross_entropy_kernel, element_mul_kernel

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576 https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2


def fused_linear_cross_entropy_forward(_input, weight, target, bias=None, ignore_index=-100, label_smoothing=0.0):
    dtype = get_autocast_gpu_dtype() if is_autocast_enabled() else _input.dtype

    # inputs have shape: BT x H
    # materialized activations will have shape: BT x V
    # the increase in memory = BT x V
    # reduction can be achieved by partitioning the number of tokens BT into smaller chunks.
    # for ex: if we were to achieve the same memory consumption as BT x H, then the chunk size should be:
    # inc_factor = (V+H-1)//H, chunk_size = (BT + inc_factor - 1)//inc_factor
    # for ex: BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = 8, chunk_size = 2048
    BT, H = _input.shape
    V = weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))  # (BT + inc_factor - 1) // inc_factor
    num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size

    grad_weight = paddle.zeros_like(weight)
    grad_input = paddle.zeros_like(_input)
    grad_bias = paddle.zeros_like(bias) if bias is not None else None
    # we use fp32 for loss accumulator
    loss_1d = paddle.zeros(
        [
            BT,
        ],
        dtype=paddle.float32,
    )

    total_n_non_ignore = (target != ignore_index).sum().item()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]  # chunk_size x H

        # when doing matmul, use the original precision
        logits_chunk = _input_chunk @ weight.t()  # chunk_size x V
        if bias is not None:
            logits_chunk = logits_chunk + bias
        target_chunk = target[start_idx:end_idx]  # chunk_size,

        n_rows = logits_chunk.shape[0]

        # unreduced loss
        loss_1d_slice = loss_1d[start_idx:end_idx]  # chunk_size,
        n_non_ignore = (target_chunk != ignore_index).sum().item()

        # when doing CE, use the upcasted precision
        logits_chunk = logits_chunk.float()

        # ensure _input and target are contiguous
        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        # Here we calculate the gradient of logits_chunk in place so we can save memory.
        cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.strides[-2],
            Y_ptr=target_chunk,
            Y_stride=target_chunk.strides[-1],  # always 1
            loss_ptr=loss_1d_slice,
            loss_stride=loss_1d_slice.strides[-1],  # always 1
            n_cols=V,
            n_non_ignore=n_non_ignore,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

        # gradient of logits_chunk is computed in-place by the above triton kernel.
        # Following HuggingFace model source code, we do the forward and backward
        # w.r.t. logits in fp32 for numerical stability especially as the num classes (vocab size) os huge.
        # (reference: https://github.com/huggingface/transformers/blob/v4.42.4/src/transformers/models/llama/modeling_llama.py#L1194)
        # Propagating to lm_head's backward, we'll switch back to the original dtype.
        logits_chunk = logits_chunk.to(dtype)

        # gradient of logits_chunk is computed in-place by the above triton kernel and is of shape: chunk_size x V
        # thus grad_input[start_idx: end_idx] should be of shape: chunk_size x H
        # additionally, since we are chunking the inputs, observe that the loss and gradients are calculated only
        # on `n_non_ignore` tokens. However, the gradient of the input should be calculated for all tokens.
        # Thus, we need an additional scaling factor of (n_non_ignore/total_n_non_ignore) to scale the gradients.
        grad_logits_chunk = logits_chunk * (n_non_ignore / total_n_non_ignore)  # chunk_size x V
        grad_input[start_idx:end_idx] = grad_logits_chunk @ weight
        grad_weight = paddle.addmm(
            input=grad_weight,
            x=logits_chunk.t(),
            y=_input_chunk,
            alpha=n_non_ignore / total_n_non_ignore,
            beta=1.0,
        )

        if bias is not None:
            grad_bias = paddle.add(
                x=grad_bias,
                y=logits_chunk.sum(axis=0) * n_non_ignore / total_n_non_ignore,
            )

    loss = paddle.sum(loss_1d) / total_n_non_ignore
    return loss, grad_input, grad_weight, grad_bias


def fused_linear_cross_entropy_backward(grad_output, grad_input, grad_weight, grad_bias):
    # If cross entropy is the last layer, grad_output is 1.0. Skip the mul to save time
    if paddle.not_equal(grad_output, paddle.to_tensor([1.0], dtype=grad_output.dtype)):
        # We use a Triton kernel instead of a PyTorch operation because modifying inputs in-place
        # for gradient storage and backward multiple times causes anomalies with PyTorch but not with Triton.
        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.strides[-2],
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

        # handle grad_weight
        V, H = grad_weight.shape
        n_rows = V

        element_mul_kernel[(n_rows,)](
            grad_weight,
            grad_weight.strides[-2],
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

        if grad_bias is not None:
            V = grad_bias.shape[0]
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_bias,
                grad_bias.strides[-1],
                grad_output,
                1,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32,
            )
    return grad_input, grad_weight, grad_bias


class FusedLinearCrossEntropyFunction(paddle.autograd.PyLayer):
    @staticmethod
    @custom_fwd
    def forward(ctx, _input, weight, target, bias=None, ignore_index=-100, label_smoothing=0.0):
        """
        Fusing the last linear layer with cross-entropy loss
            Reference: https://github.com/mgmalek/efficient_cross_entropy

        Handle the forward and backward pass of the final linear layer via cross-entropy loss by avoiding the
        materialization of the large logits tensor. Since Cross Entropy Loss is the last layer, we can compute the
        gradient at the forward pass. By doing so, we don't have to store the _input and target for the backward pass.

        _input: (B*T, H) where B is batch size, T is sequence length, H is hidden dimension. target: (B*T) where each
        value is in [0, V-1] weight: (V, H) where V is the number of classes bias: (V) where V is the number of classes
        ignore_index: the index to ignore in the target
        """
        loss, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward(
            _input, weight, target, bias, ignore_index, label_smoothing
        )
        # downcast to dtype and store for backward
        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach(),
            grad_bias.detach() if bias is not None else None,
        )
        return loss

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        (grad_input, grad_weight, grad_bias) = ctx.saved_tensor()
        grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )
        return (grad_input, grad_weight, None, grad_bias, None)
