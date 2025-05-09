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

# Notice: this kernel is especially implemented for the segment mean operation for a varlen qkv tensor.
# for example, the k tensor is: [total_seqlen, num_head, head_dim],
# where total_seqlen = seqlen 1 + seqlen 2 + ... + seqlen n.
# So the segment mean triton kernel will do mean operation along the **seqlen** dim.
# It will finally generate a `[bsz, num_head, head_dim]` shape-like result,
# as the result of mean value of each seqlen segment.

import paddle
import triton.language as tl
from paddle import _C_ops
from paddle.base.framework import OpProtoHolder
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode

from paddlenlp.ops.triton_ops.triton_utils import (
    get_dtype_str,
    paddle_use_triton,
    rendering_common_template,
)


@paddle_use_triton(key=["num_heads", "head_dim"])
def segmented_mean_reduce_kernel(
    input_ptr,
    output_ptr,
    cu_seqlen_ptr,
    num_batches,
    num_heads,
    head_dim,
    input_stride_seq,
    input_stride_head,
    output_stride_batch,
    output_stride_head,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_HEAD: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_offset = tl.program_id(1) * BLOCK_SIZE_HEAD
    dim_offset = tl.program_id(2) * BLOCK_SIZE_DIM

    # 获取当前 segment 的 range
    seq_start = tl.load(cu_seqlen_ptr + batch_idx)
    seq_end = tl.load(cu_seqlen_ptr + batch_idx + 1)
    seq_len = seq_end - seq_start

    # head 和 dim 的实际索引（block中相对位置）
    head_idx = head_offset + tl.arange(0, BLOCK_SIZE_HEAD)
    dim_idx = dim_offset + tl.arange(0, BLOCK_SIZE_DIM)
    mask_head = head_idx < num_heads
    mask_dim = dim_idx < head_dim

    # 初始化累加器（float32 精度）
    acc = tl.zeros((BLOCK_SIZE_HEAD, BLOCK_SIZE_DIM), dtype=tl.float32)

    for seq_offset in range(0, seq_len, BLOCK_SIZE_SEQ):
        local_seq_idx = tl.arange(0, BLOCK_SIZE_SEQ)
        mask_seq = local_seq_idx < (seq_len - seq_offset)
        global_seq = seq_start + seq_offset + local_seq_idx
        # shape: [BLOCK_SIZE_SEQ, BLOCK_SIZE_HEAD, BLOCK_SIZE_DIM]
        input_ptrs = (
            input_ptr
            + global_seq[:, None, None] * input_stride_seq
            + head_idx[None, :, None] * input_stride_head
            + dim_idx[None, None, :]
        )

        # 加载输入，注意输入 dtype 指明 float16 以避免不必要转换
        x = tl.load(
            input_ptrs, mask=mask_seq[:, None, None] & mask_head[None, :, None] & mask_dim[None, None, :], other=0.0
        ).to(tl.float32)

        acc += tl.sum(x, axis=0)  # reduce over seq axis

    mean = acc / seq_len

    # 构造输出地址
    output_ptrs = (
        output_ptr + batch_idx * output_stride_batch + head_idx[:, None] * output_stride_head + dim_idx[None, :]
    )

    tl.store(output_ptrs, mean.to(input_ptr.dtype.element_ty), mask=mask_head[:, None] & mask_dim[None, :])


def segment_mean(
    x: paddle.Tensor, cu_seqlen: paddle.Tensor  # [total_seqlen, num_heads, head_dim]  # [batch_size + 1]
):
    """
    Examples:
        import paddle
        from paddlenlp.ops.triton_ops.segment_mean import segment_mean

        cu_seqlens = [0, 1024, 2048, 4096]
        total_seqlen = 4096
        num_head = 24
        head_dim = 128
        k = paddle.randn([total_seqlen, num_head, head_dim], dtype="float16")
        cu_seqlen = paddle.to_tensor(cu_seqlens, paddle.int32)
        km = segment_mean(k, cu_seqlen)
    """
    num_batches = cu_seqlen.shape[0] - 1

    num_heads = x.shape[1]
    head_dim = x.shape[2]

    # 计算必要的strides
    input_stride_seq = num_heads * head_dim
    input_stride_head = head_dim

    output_stride_batch = num_heads * head_dim
    output_stride_head = head_dim
    prepare_attr_for_triton_kernel = """
    const int num_batches = cu_seqlen.shape()[0] - 1;
    const int num_heads = x.shape()[1];
    const int head_dim = x.shape()[2];
    int input_stride_seq = num_heads * head_dim;
    int input_stride_head = head_dim;
    int output_stride_batch = num_heads * head_dim;
    int output_stride_head = head_dim;
    paddle::Tensor output_tensor = paddle::empty({num_batches, num_heads, head_dim}, x.dtype(), x.place());
"""
    op_name = "triton_segment_mean"
    op_name += get_dtype_str(x.dtype)

    # auto-tuning
    segment_mean_configs = []
    segment_mean_configs.append(
        {"BLOCK_SIZE_SEQ": 128, "BLOCK_SIZE_HEAD": 4, "BLOCK_SIZE_DIM": 64, "num_stages": 2, "num_warps": 4}
    )
    segment_mean_configs.append(
        {"BLOCK_SIZE_SEQ": 256, "BLOCK_SIZE_HEAD": 4, "BLOCK_SIZE_DIM": 64, "num_stages": 2, "num_warps": 4}
    )
    segment_mean_configs.append(
        {"BLOCK_SIZE_SEQ": 512, "BLOCK_SIZE_HEAD": 8, "BLOCK_SIZE_DIM": 64, "num_stages": 2, "num_warps": 8}
    )
    segment_mean_configs.append(
        {"BLOCK_SIZE_SEQ": 256, "BLOCK_SIZE_HEAD": 8, "BLOCK_SIZE_DIM": 128, "num_stages": 2, "num_warps": 4}
    )

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        Output = paddle.empty([num_batches, num_heads, head_dim], dtype=x.dtype)
        prepare_ptr_for_triton_kernel = """
    // prepare tensor
    CUdeviceptr input_ptrs[3] = {
        get_tensor_ptr(x),
        get_tensor_ptr(output_tensor),
        get_tensor_ptr(cu_seqlen)
    };
"""
        return_tensor_names = "output_tensor"
        template_used = rendering_common_template(
            segment_mean,
            prepare_attr_for_triton_kernel,
            prepare_ptr_for_triton_kernel,
            return_tensor_names,
        )

        # 确定kernel配置
        grid = (
            "num_batches",
            "(num_heads + BLOCK_SIZE_HEAD - 1) / BLOCK_SIZE_HEAD",
            "(head_dim + BLOCK_SIZE_DIM - 1) / BLOCK_SIZE_DIM",
        )

        # 调用kernel
        segmented_mean_reduce_kernel[(op_name, template_used, grid, segment_mean_configs)](
            input_ptr=x,
            output_ptr=Output,
            cu_seqlen_ptr=cu_seqlen,
            num_batches=-1,
            num_heads=num_heads,
            head_dim=head_dim,
            input_stride_seq=input_stride_seq,
            input_stride_head=input_stride_head,
            output_stride_batch=output_stride_batch,
            output_stride_head=output_stride_head,
        )

    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(op_name, x, cu_seqlen)
        return outs[0]
    else:
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "x": x,
            "cu_seqlen_tensor": cu_seqlen,
        }
        output = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(type=op_name, inputs=inputs, outputs={"output_tensor": output})
        return output
