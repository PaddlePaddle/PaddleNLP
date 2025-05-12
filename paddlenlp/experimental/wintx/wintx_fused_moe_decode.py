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
import triton.language as tl
from paddle import _C_ops
from paddle.base.framework import OpProtoHolder
from paddle.framework import in_dynamic_or_pir_mode

from paddlenlp.ops.triton_ops.triton_utils import (
    get_dtype_str,
    paddle_use_triton,
    rendering_common_template,
)

__all__ = ["fused_moe_wintx_decode_wint2_75", "fused_moe_wintx_decode_wint2_5"]
BLOCK_SIZE_M = 16


def invoke_fused_moe_kernel(
    A,
    B,
    C,
    B_scale,
    topk_weights,
    topk_ids,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    bit_shift,
    mul_routed_weight=False,
    top_k=-1,
    group_size=-1,
    ppack_num=3,
    ww_mask=0xF,
    ss_mask=0xF,
    bbzp=8,
):

    # bit_shift = paddle.to_tensor([4,2,0],dtype='int8')

    KK = A.shape[-1]
    NN = B.shape[-1]
    EEM = sorted_token_ids.shape[0]
    sstride_am, sstride_ak = A.shape[1], 1
    sstride_be, sstride_bk, sstride_bn = B.shape[1] * B.shape[2], B.shape[2], 1
    sstride_cm, sstride_cn = C.shape[-1], 1
    sstride_bse, sstride_bsk, sstride_bsn = B_scale.shape[1], 1, 1
    nnum_valid_tokens = topk_ids.numel().tolist()

    prepare_attr_for_triton_kernel = """
        auto N = B.shape()[2];
        auto K = A.shape()[1];
        auto EM = sorted_token_ids.shape()[0];
        auto num_valid_tokens = (topk_ids.shape()[0]) * (topk_ids.shape()[1]);
        auto stride_am = A.strides()[0];
        auto stride_ak = A.strides()[1];
        auto stride_be = B.strides()[0];
        auto stride_bk = B.strides()[1];
        auto stride_bn = B.strides()[2];
        auto stride_cm = C.strides()[1];
        auto stride_cn = C.strides()[2];

        auto stride_bse = B_scale.strides()[0];
        auto stride_bsk = B_scale.strides()[1];
        auto stride_bsn = 1;

        auto pack_num = ppack_num;
        auto w_mask = ww_mask;
        auto s_mask = ss_mask;
        auto bzp = bbzp;
    """

    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 4,
    }
    configs = []

    configs.append(dict(config))

    op_name = f"fused_moe_paddle_wintx_{ppack_num}_{ww_mask}_{ss_mask}_{bbzp}"
    op_name += f"{get_dtype_str(A.dtype)}"
    op_name += f"{B.shape[0]}"
    op_name += f"{B.shape[1]}"
    op_name += f"{B.shape[2]}"

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        prepare_ptr_for_triton_kernel = """
            CUdeviceptr input_ptrs[9] = {
                get_tensor_ptr(A),
                get_tensor_ptr(B),
                get_tensor_ptr(C),
                get_tensor_ptr(B_scale),
                get_tensor_ptr(topk_weights),
                get_tensor_ptr(sorted_token_ids),
                get_tensor_ptr(expert_ids),
                get_tensor_ptr(num_tokens_post_padded),
                get_tensor_ptr(bit_shift),
            };
            """
        template_used = rendering_common_template(
            invoke_fused_moe_kernel,
            prepare_attr_for_triton_kernel,
            prepare_ptr_for_triton_kernel,
        )
        grid = ("(EM+BLOCK_SIZE_M-1)/BLOCK_SIZE_M * ((N+BLOCK_SIZE_N-1)/BLOCK_SIZE_N)",)

        fused_moe_decode_kernel_paddle[(op_name, template_used, grid, configs)](
            A,
            B,
            C,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            bit_shift,
            NN,
            KK,
            EEM,
            nnum_valid_tokens,
            sstride_am,
            sstride_ak,
            sstride_be,
            sstride_bk,
            sstride_bn,
            sstride_cm,
            sstride_cn,
            sstride_bse,
            sstride_bsk,
            sstride_bsn,
            MUL_ROUTED_WEIGHT=(int)(mul_routed_weight),
            top_k=top_k,
            BLOCK_SIZE_K=group_size,
            pack_num=ppack_num,
            w_mask=ww_mask,
            s_mask=ss_mask,
            bzp=bbzp,
        )
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            op_name,
            A,
            B,
            C,
            B_scale,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            bit_shift,
            mul_routed_weight,
            top_k,
            group_size,
            ppack_num,
            ww_mask,
            ss_mask,
            bbzp,
        )
        return outs[0]


@paddle_use_triton(
    key=["1"],
)
def fused_moe_decode_kernel_paddle(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    bs_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    bit_shift_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    # group_n: tl.constexpr,
    # group_size: tl.constexpr, # forced equal to BK
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    pack_num: tl.constexpr = 7,
    w_mask: tl.constexpr = 0x7,
    s_mask: tl.constexpr = 0xF,
    bzp: tl.constexpr = 4,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """

    real_k_size: tl.constexpr = (BLOCK_SIZE_K - 1) // pack_num + 1

    pid = tl.program_id(axis=0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # maybe more efficient by set bf16
    compute_type = c_ptr.dtype.element_ty

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)

    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] // pack_num * stride_bk + offs_bn[None, :] * stride_bn)

    # maybe more efficient by eliminate load process
    b_shift_bits = tl.load(bit_shift_ptr + offs_k[:, None] % pack_num)

    bs_ptrs = bs_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    super_bs = tl.load(bs_ptrs)  # super scale
    scale_idx = tl.arange(0, BLOCK_SIZE_K)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None],
            other=0.0,
        )
        b = tl.load(b_ptrs)

        # maybe more efficent
        bs = tl.sum(tl.where(scale_idx[:, None] == BLOCK_SIZE_K - 1, b, 0), 0)
        bs = (bs & s_mask)[None, :] * super_bs
        b = (((b >> b_shift_bits) & w_mask) - bzp) * bs
        accumulator += tl.dot(a, b.to(a.dtype))

        b_ptrs += real_k_size * stride_bk
        a_ptrs += BLOCK_SIZE_K * stride_ak

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def fused_moe_wintx_decode_impl(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    # inplace: bool = False,
    w1_scale=None,
    w2_scale=None,
    group_size=64,
    bit="wint2.75",
):

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert group_size > 0, "Group size must be greater than 0"

    num_tokens, K = hidden_states.shape
    E, _, N = w1.shape
    M = num_tokens

    if group_size < 0:
        group_size = K // w1_scale.shape[1]

    top_k = topk_ids.shape[1]

    intermediate_cache1 = paddle.zeros(
        [M, top_k, N],
        dtype=hidden_states.dtype,
    )
    intermediate_cache2 = paddle.zeros(
        (M * top_k, N // 2),
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = paddle.zeros(
        (M, top_k, K),
        dtype=hidden_states.dtype,
    )

    from paddlenlp_ops import preprocess_for_moe

    sorted_token_ids, expert_ids, num_tokens_post_padded = preprocess_for_moe(topk_ids, E, BLOCK_SIZE_M)

    if bit == "wint2.75":
        bit_shift = paddle.to_tensor([4, 2, 0], dtype="int8")
        ppack_num = 3
        ww_mask = 0xF
        ss_mask = 0xF
        bbzp = 8
    elif bit == "wint2.5":
        ppack_num = 7
        ww_mask = 0x7
        ss_mask = 0x1FFF
        bbzp = 4
        bit_shift = paddle.to_tensor([13, 11, 9, 6, 4, 2, 0], dtype="int16")

    invoke_fused_moe_kernel(
        A=hidden_states,
        B=w1,
        C=intermediate_cache1,
        B_scale=w1_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        bit_shift=bit_shift,
        mul_routed_weight=False,
        top_k=top_k,
        group_size=group_size,
        ppack_num=ppack_num,
        ww_mask=ww_mask,
        ss_mask=ss_mask,
        bbzp=bbzp,
    )

    intermediate_cache2 = paddle.incubate.nn.functional.swiglu(intermediate_cache1.reshape([-1, N]))

    invoke_fused_moe_kernel(
        A=intermediate_cache2,
        B=w2,
        C=intermediate_cache3,
        B_scale=w2_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        bit_shift=bit_shift,
        mul_routed_weight=True,
        top_k=1,
        group_size=group_size,
        ppack_num=ppack_num,
        ww_mask=ww_mask,
        ss_mask=ss_mask,
        bbzp=bbzp,
    )

    out_hidden_states = paddle.sum(intermediate_cache3, axis=1)

    del intermediate_cache1, intermediate_cache2, intermediate_cache3
    del sorted_token_ids, expert_ids, num_tokens_post_padded

    return out_hidden_states


def fused_moe_wintx_decode_wint2_75(
    hidden_states,
    w1,
    w2,
    scores,
    topk: int,
    w1_scale=None,
    w2_scale=None,
):

    topk_weights, topk_ids = paddle.topk(scores, k=topk, axis=-1, sorted=False)

    return fused_moe_wintx_decode_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        # inplace: bool = False,
        w1_scale,
        w2_scale,
        bit="wint2.75",
    )


def fused_moe_wintx_decode_wint2_5(
    hidden_states,
    w1,
    w2,
    scores,
    topk: int,
    w1_scale=None,
    w2_scale=None,
):

    topk_weights, topk_ids = paddle.topk(scores, k=topk, axis=-1, sorted=False)

    return fused_moe_wintx_decode_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        # inplace: bool = False,
        w1_scale,
        w2_scale,
        bit="wint2.5",
    )
