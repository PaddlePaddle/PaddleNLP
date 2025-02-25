import paddle
import paddlenlp_ops

from typing import Optional

def per_block_int8(
    q: paddle.Tensor, 
    k: paddle.Tensor, 
    km: Optional[paddle.Tensor] = None, 
    BLKQ: int =128,
    BLKK: int =64,
    sm_scale: Optional[float] = None, 
    tensor_layout: str ="HND"
):
    q_int8 = paddle.empty(q.shape, dtype=paddle.int8)
    k_int8 = paddle.empty(k.shape, dtype=paddle.int8)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    _tensor_layout = 0 if tensor_layout == "NHD" else 1

    q_scale = paddle.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ), dtype=paddle.float32)
    k_scale = paddle.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), dtype=paddle.float32)

    if sm_scale is None:
        sm_scale = head_dim**-0.5
    
    sm_scale *= 1.44269504

    paddlenlp_ops.quant_per_block_int8_cuda(q, q_int8, q_scale, sm_scale, BLKQ, _tensor_layout)
    if km is not None:
        km = km.squeeze(1) if _tensor_layout == 0 else km.squeeze(2)
        paddlenlp_ops.quant_per_block_int8_fuse_sub_mean_cuda(k, km, k_int8, k_scale, BLKK, _tensor_layout)
    else:
        paddlenlp_ops.quant_per_block_int8_cuda(k, k_int8, k_scale, BLKK, _tensor_layout)

    return q_int8, q_scale, k_int8, k_scale


def per_warp_int8(
    q: paddle.Tensor, 
    k: paddle.Tensor, 
    km: Optional[paddle.Tensor] = None, 
    BLKQ: int =128,
    WARPQ: int =32,
    BLKK: int =64,
    tensor_layout: str ="HND"
):
    q_int8 = paddle.empty(shape=q.shape, dtype=paddle.int8)
    k_int8 = paddle.empty(shape=k.shape, dtype=paddle.int8)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    
    _tensor_layout = 0 if tensor_layout == "NHD" else 1

    q_scale = paddle.empty((b, h_qo, ((qo_len + BLKQ - 1) // BLKQ) * (BLKQ // WARPQ)), dtype=paddle.float32)
    k_scale = paddle.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), dtype=paddle.float32)

    paddlenlp_ops.quant_per_warp_int8_cuda(q, q_int8, q_scale, BLKQ, WARPQ, _tensor_layout)

    if km is not None:
        km = km.squeeze(1) if _tensor_layout == 0 else km.squeeze(2)
        paddlenlp_ops.quant_per_block_int8_fuse_sub_mean_cuda(k, km, k_int8, k_scale, BLKK, _tensor_layout)
    else:
        paddlenlp_ops.quant_per_block_int8_cuda(k, k_int8, k_scale, BLKK, _tensor_layout)
    
    return q_int8, q_scale, k_int8, k_scale


def per_channel_fp8(
    v: paddle.Tensor,
    tensor_layout: str ="NHD",
    scale_max: float = 448.0,
    smooth_v: bool = True
):
    _tensor_layout = 0 if tensor_layout == "NHD" else 1

    if tensor_layout == "HND":
        b, h_kv, kv_len, head_dim = v.shape
        padded_len = (kv_len + 63) // 64 * 64
        v_transposed_permutted = paddle.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype)

    elif tensor_layout == "NHD":
        b, kv_len, h_kv, head_dim = v.shape
        padded_len = (kv_len + 63) // 64 * 64
        v_transposed_permutted = paddle.empty((b, head_dim, h_kv, padded_len), dtype=v.dtype)
    paddlenlp_ops.transpose_pad_permute_cuda(v, v_transposed_permutted, _tensor_layout)

    v_fp8 = paddle.empty(v_transposed_permutted.shape, dtype=paddle.float8_e4m3fn)

    v_scale = paddle.empty((b, h_kv, head_dim), dtype=paddle.float32)
    vm = paddle.empty((b, h_kv, head_dim), dtype=paddle.float32)

    if smooth_v:
        paddlenlp_ops.mean_scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, vm, v_scale, kv_len, scale_max, _tensor_layout)
        return v_fp8, v_scale, vm
    else:
        paddlenlp_ops.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, scale_max, _tensor_layout)
        return v_fp8, v_scale, None
    

def sub_mean(
    v: paddle.Tensor, 
    tensor_layout: str ="HND"
):
    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    vm = v.mean(dim=1 if _tensor_layout == 0 else 2)

    v_smoothed = paddle.empty(v.shape, dtype=paddle.float16)
    
    # subtract mean and store the result as fp16
    paddlenlp_ops.sub_mean_cuda(v, vm, v_smoothed, _tensor_layout)

    return v_smoothed, vm