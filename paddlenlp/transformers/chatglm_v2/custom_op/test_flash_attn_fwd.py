import math
import paddle
from paddle.utils.cpp_extension import CUDAExtension, setup, load
from flash_atten2 import flash_attn_fwd

# setup(
#     name='custom_setup_ops',
    
#     ext_modules=CUDAExtension(
#         sources=['flash_attn_fwd.cu', 'flash_attention/flash_fwd_sm80.cu']
#     ),
#     include_dirs=['cutlass/include'],
#     extra_compile_args={"nvcc": "-std=c++17"},  
# )


def test_flash_attn_fwd():
    batch_size = 1
    seqlen_q = 16
    seqlen_k = 16
    num_heads = num_heads_k = 8
    head_size = 32
    softmax_scale = 1. / math.sqrt(head_size)
    is_causal = False

    q = paddle.randn([batch_size, seqlen_q, num_heads, head_size], dtype="float16")
    k = paddle.randn([batch_size, seqlen_k, num_heads_k, head_size], dtype="float16")
    v = paddle.randn([batch_size, seqlen_k, num_heads_k, head_size], dtype="float16")
    out = flash_attn_fwd(q, k, v, softmax_scale, is_causal)
    print(out)

# test_flash_attn_fwd()


def test_flash_attn_varlen_fwd():
    cu_seqlens_q = [3, 8, 5]
    cu_seqlens_k = [3, 8, 5]
    total_q = sum(cu_seqlens_q)
    total_k = sum(cu_seqlens_k)
    max_seqlen_q = 10
    max_seqlen_k = 10
    num_heads = num_heads_k = 8
    head_size = 32
    softmax_scale = 1. / math.sqrt(head_size)
    zero_tensors = False
    is_causal = False
    q = paddle.randn([total_q, num_heads, head_size], dtype="float16")
    k = paddle.randn([total_k, num_heads_k, head_size], dtype="float16")
    v = paddle.randn([total_k, num_heads_k, head_size], dtype="float16")
    cu_seqlens_q = paddle.to_tensor(cu_seqlens_q, dtype = "int32")
    cu_seqlens_k = paddle.to_tensor(cu_seqlens_k, dtype = "int32")
    out = custom_op_module.flash_attn_varlen_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, zero_tensors, is_causal)
    print(out)

#test_flash_attn_varlen_fwd()

def test_flash_attn_fwd_1():
    batch_size = 6
    seqlen_q = 1350
    seqlen_k = 1350
    num_heads = num_heads_k = 12
    head_size = 64
    softmax_scale = 1. / math.sqrt(head_size)
    is_causal = False

    count=100
    while count>0:
        q = paddle.randn([batch_size, seqlen_q, num_heads, head_size], dtype="float16")
        k = paddle.randn([batch_size, seqlen_k, num_heads_k, head_size], dtype="float16")
        v = paddle.randn([batch_size, seqlen_k, num_heads_k, head_size], dtype="float16")
        out = flash_attn_fwd(q, k, v, softmax_scale, is_causal)
        count=count-1
        if count==0:
            print(out)


test_flash_attn_fwd_1()