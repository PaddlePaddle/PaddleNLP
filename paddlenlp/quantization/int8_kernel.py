import paddle
import triton
from typing import Tuple, List, Optional
import triton.language as tl

@triton.jit
def _per_token_group_quant_int8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    y_stride,
    N,
    eps,
    int8_min,
    int8_max,
    BLOCK: tl.constexpr,
):
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / int8_max
    y_q = tl.clamp(y / y_s, int8_min, int8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)

@triton.jit
def _w8a8_block_int8_matmul(
    A_ptr, B_ptr, C_ptr,
    A_scale, B_scale,
    M, N, K,
    group_n, group_k,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_As_m, stride_As_k,
    stride_Bs_k, stride_Bs_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    num_warps: tl.constexpr,
    num_states: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = A_scale + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = B_scale + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b).to(tl.float32) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C_ptr.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C_ptr.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
    


# 量化函数
def per_token_group_quant_int8(
    x: paddle.Tensor,
    group_size = 2,
    eps: float = 1e-10,
    dtype: paddle.dtype = paddle.int8,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    对输入张量 x 进行分组量化，返回量化后的张量和量化 scale。

    Args:
        x (paddle.Tensor): 输入张量，形状为 [B, ..., input_dim]
        group_size (int): 分组大小
        eps (float): 避免除零的小常数
        dtype (paddle.dtype): 量化类型，默认为 int8

    Returns:
        Tuple[paddle.Tensor, paddle.Tensor]: 量化后的张量和量化 scale
    """
    assert x.ndim >= 2, "输入张量 x 的维度必须至少为 2"
    assert group_size > 0, "分组大小必须大于 0"

    assert x.shape[-1] % group_size == 0, "输入张量 x 的最后一个维度必须能被分组大小整除"
    assert x.is_contiguous(), "输入张量 x 必须是连续的"
    
    iinfo = paddle.iinfo(dtype)
    q_min, q_max = iinfo.min, iinfo.max

    x_q = paddle.empty_like(x, dtype=dtype).to(x.place)
    M = x.numel() // group_size
    N = group_size
    x_s_shape = x.shape[:-1] + [x.shape[-1] // group_size, ]
    x_s = paddle.empty(
        x_s_shape,
        dtype=paddle.float16,
    ).to(x.place)

    
    BLOCK = triton.next_power_of_2(N) # 获取大于N的2的幂
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    _per_token_group_quant_int8[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        N,
        eps,
        int8_min=q_min,
        int8_max=q_max,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return x_q, x_s  

# 去量化函数
def block_dequant(
    x_q_block: paddle.Tensor,
    x_s: paddle.Tensor,
    block_size: list[int],
) -> paddle.Tensor:
    block_n, block_k = block_size[0], block_size[1]
    n, k = x_q_block.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    assert n_tiles == x_s.shape[0]
    assert k_tiles == x_s.shape[1]

    x_dq_block = x_q_block.to(paddle.float32)

    for i in range(k_tiles):
        for j in range(n_tiles):
            row_start = j * block_n
            row_end = min((j + 1) * block_n, n)
            col_start = i * block_k
            col_end = min((i + 1) * block_k, k)
            x_dq_block[row_start:row_end, col_start:col_end] *= x_s[j, i]

    return x_dq_block


def w8a8_block_int8_matmul(
    A: paddle.Tensor,
    B: paddle.Tensor,
    A_scale: paddle.Tensor,
    B_scale: paddle.Tensor,
    block_size: list[int],
    output_dtype: paddle.dtype = paddle.float32,
) -> paddle.Tensor:
    """
    Perform a block-wise INT8 matrix multiplication with W8A8 quantization.
    
    Args:
        A: Input tensor A, shape [M, K] [7, 896]
        B: Input tensor B, shape [N, K] [896, 896]
        A_scale: Scale tensor for A, shape [M//block_m, K//block_k] [7, 14]
        B_scale: Scale tensor for B, shape [K//block_k, N//block_n] [14, 896]
        block_size: Block size for quantization, list of two integers [M_block, K_block] [1, 64]
        output_dtype: Output data type, default is float32 paddle.float16

    Returns:
        paddle.Tensor: Output tensor, shape [B, M, N]
    """
    assert len(block_size) == 2, "block_size 必须是一个包含两个整数的列表"
    block_n, block_k = block_size

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == A_scale.shape[:-1] and A.is_contiguous()

    assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
    # M = A.numel() // A.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and B_scale.ndim == 2
    M, K = A.shape
    N = B.shape[0]


    assert triton.cdiv(N, block_n) == B_scale.shape[0]
    assert triton.cdiv(K, block_k) == B_scale.shape[1]

    C_shape = (M, N)
    # C = A.new_empty(C_shape, dtype=output_dtype)
    C = paddle.empty(C_shape, dtype=output_dtype)

    config = {
        "BLOCK_SIZE_M": triton.next_power_of_2(64),
        "BLOCK_SIZE_N": triton.next_power_of_2(block_n),
        "BLOCK_SIZE_K": triton.next_power_of_2(block_k),
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_states": 3
    }

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        
    A_1, A_2 = A.strides[-2], A.strides[-1]
    _w8a8_block_int8_matmul[grid](
        A, B, C,
        A_scale, B_scale,
        M, N, K,
        block_n, block_k,
        A_1, A_2,
        B.strides[1], B.strides[0],
        C.strides[-2], C.strides[-1],
        A_scale.strides[-2], A_scale.strides[-1],
        B_scale.strides[1], B_scale.strides[0],
        **config,
    )

    return C

def create_int8_parameter(
    layer: paddle.nn.Layer,
    name: str,
    shape: List[int],
    low: int = -127,
    high: int = 128,
    initializer: Optional[callable] = None,
) -> paddle.Tensor:
    """
    创建并注册一个 int8 张量到层中，作为不可训练的权重。
    
    Args:
        layer (paddle.nn.Layer): 目标层。
        name (str): 参数名称。
        shape (List[int]): 张量形状。
        low (int): 随机初始化的下界，默认 -127。
        high (int): 随机初始化的上界，默认 128。
        initializer (callable, optional): 自定义初始化函数，接受 shape 参数返回 int8 张量。
    
    Returns:
        paddle.Tensor: 创建的 int8 张量。
    """
    # 如果提供了自定义初始化器，则使用它
    if initializer is not None:
        weight_tensor = initializer(shape)
        if not isinstance(weight_tensor, paddle.Tensor) or weight_tensor.dtype != paddle.int8:
            raise ValueError("Initializer must return a paddle.Tensor with dtype=int8")
    else:
        # 默认使用随机初始化：先生成 int32 张量，再转换为 int8
        weight_tensor = paddle.randint(low=low, high=high, shape=shape, dtype=paddle.int32)
        weight_tensor = weight_tensor.astype(paddle.int8)

    return weight_tensor
