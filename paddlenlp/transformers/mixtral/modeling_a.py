
import triton
import triton.language as tl


@triton.jit()
def col_major(pid,
              m, n, num_tokens_post_padded,
              block_m: tl.constexpr, block_n: tl.constexpr):
    
    grid_m = tl.cdiv(m, block_m)    
    grid_n = tl.cdiv(n, block_n)

    pid_m_max = (num_tokens_post_padded // block_m) * 2

    pid_m = (pid % grid_n) % pid_m_max
    pid_n = pid // grid_m

    return pid_m, pid_n


@triton.jit()
def fused_moe_kernel_splitk(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_weight,
    stride_token_id,
    # Meta-parameters
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    split_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):  
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    
    # Scheduling Problem
    pid = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    
    # num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    # tl.device_print("num_tokens_post_padded", num_tokens_post_padded)

    # print("num_tokens_post_padded: ", num_tokens_post_padded)

    # pid_m, pid_n = col_major(pid,
    #                          EM, N, num_tokens_post_padded,
    #                          block_m, block_n)

    # pid_m, pid_n = swizzle_tile(pid,
    #                             EM, N,
    #                             block_m, block_n, group_m)
    
    total_blocks_k = tl.cdiv(K, block_k*split_k)
    
    # num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    # if pid_m * block_m >= num_tokens_post_padded:
    #     return
    
    grid_n = tl.cdiv(N, block_n)
    grid_m = tl.cdiv(EM, block_m)
    # pid_m = pid // (grid_n)
    # pid_n = pid % (grid_n)

    pid_m = pid % (grid_m)
    pid_n = pid // (grid_m)


    offs_token_id = pid_m * block_m + tl.arange(0, block_m)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
   
    offs_bn = (pid_n * block_n + tl.arange(0, block_n)) % N
    offs_k = pid_k*block_k + tl.arange(0, block_k)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    
    off_experts = tl.load(expert_ids_ptr + pid_m * block_m)
    
    if off_experts < 0:
        return 
    
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k in range(0, total_blocks_k):
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None],
                    other=0.0)
        b = tl.load(b_ptrs)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += block_k * stride_ak * split_k
        b_ptrs += block_k * stride_bk * split_k

    if MUL_ROUTED_WEIGHT == 1:
        moe_weight = tl.load(topk_weights_ptr + offs_token * stride_weight,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * block_n + tl.arange(0, block_n)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.atomic_add(c_ptrs, accumulator, mask=c_mask)
