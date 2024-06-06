

import paddle
import paddle.nn.functional as F
k = 2
paddle.seed(123)
# def basic_moe_fc(activations, expert_for_row, weights, scales=None, biases=None):   
#     res = paddle.zeros([activations.shape[0], weights.shape[-1]], dtype="float16")
#     for i, w in enumerate(weights):
#         _m = (expert_for_row == i)
#         ins = activations[_m]
#         if ins.shape[0]:
#             # # print(f'expert-{i} input={ins.shape} {_m}')
#             out = F.linear(ins, w)
#             res[_m] = out
#     return res


def basic_moe_fc_geglu(activations, 
                       expert_for_row, weights_in, weights_out, 
                       bias_in, bias_out, 
                       scales=None, biases=None, expert_cap=None):   
    res = paddle.zeros([activations.shape[0], weights_out.shape[-1]], dtype="float16")
    for i, (wi, wo, bi, bo) in enumerate(zip(weights_in, weights_out, bias_in, bias_out)):
        _m = (expert_for_row == i)
        ins = activations[_m]
        if ins.shape[0]:
            # print(f'expert-{i} input={ins.shape} ')
            x = ins
            x = F.linear(x, wi) + bi
            x, gate = x.chunk(2, axis=-1)
            x = F.silu(x) * gate
            out = F.linear(x, wo)
            out = out + bo
            res[_m] = out
    return res

# k就是topk的意思
def run_ref_moe(input_dict, k):
    gates = F.softmax(input_dict["gating_output"], axis=-1, dtype='float32') #.cast(input_dict["gating_output"].dtype)
    expert_scales, experts_for_row = paddle.topk(gates, k, axis=-1)
    denorm = paddle.clip(expert_scales.sum(-1, keepdim=True), min=1e-6)
    expert_scales = expert_scales / denorm  # [S,2]
    
    current_experts_for_row = experts_for_row[:,:k].reshape([-1])

    moe_fc_1_result = basic_moe_fc_geglu(input_dict["input_activations"].repeat_interleave(k, axis=0).cast("float16"), 
                                         current_experts_for_row, 
                                         input_dict["fc1_expert_weights_for_ref"], 
                                         input_dict["fc2_expert_weights_for_ref"],
                                         input_dict["fc1_expert_bias_for_ref"], 
                                         input_dict["fc2_expert_bias_for_ref"],
                                         expert_cap=int(2*128/64))

    moe_fc_2_result = moe_fc_1_result.reshape([-1, k, moe_fc_1_result.shape[-1]])
    return paddle.matmul(expert_scales.unsqueeze(1).cast('float16'), moe_fc_2_result).squeeze(1)

input_dict = {}
import numpy as np

seq_len = 2048
hidden_size = 4096
intersize = 14336
expert = 8



gate_weight = paddle.randn([hidden_size, expert], dtype="float32") * 0.1

input_dict["input_activations"] = paddle.randn([seq_len, hidden_size], dtype="float32") * 0.1
input_dict["gating_output"] = paddle.matmul(input_dict["input_activations"], gate_weight)


input_dict["fc1_expert_weights_for_ref"] = paddle.randn([expert, hidden_size, intersize * 2], dtype="float16") * 0.1
input_dict["fc2_expert_weights_for_ref"] = paddle.randn([expert, intersize, hidden_size], dtype="float16") * 0.1
input_dict["fc1_expert_bias_for_ref"] = paddle.zeros([expert, intersize * 2], dtype="float16") * 0.1
input_dict["fc2_expert_bias_for_ref"] = paddle.zeros([expert, hidden_size], dtype="float16") * 0.1

# print(" input_activations fff", input_dict["input_activations"].cast("float16"))
# print(" gating_output xxxx", input_dict["gating_output"])

import time
for i in range(10):
    start = time.perf_counter()
    aaa = run_ref_moe(input_dict, 2)
    paddle.device.cuda.synchronize()

    hf_cost = (time.perf_counter() - start) * 1000

    print("Speed Paddle1111:", hf_cost)




from paddle.incubate.nn.functional import fused_moe


for i in range(10):
    start = time.perf_counter()
    paddle.device.cuda.synchronize()

    
    bbb = fused_moe(
        input_dict["input_activations"].cast("float16"),
        gate_weight.cast("float32"),
        input_dict["fc1_expert_weights_for_ref"],
        input_dict["fc1_expert_bias_for_ref"],
        input_dict["fc1_expert_bias_for_ref"],
        input_dict["fc2_expert_weights_for_ref"],
        input_dict["fc2_expert_bias_for_ref"],
        input_dict["fc2_expert_bias_for_ref"],
    )
    paddle.device.cuda.synchronize()
    
    hf_cost = (time.perf_counter() - start) * 1000

    print("Speed fused_moe:", hf_cost)

from triton_ops import triton_moe_preposs

# a,b,c = input_dict["fc1_expert_weights_for_ref"].shape
# d = input_dict["fc1_expert_weights_for_ref"]
# d = d.transpose([0,2,1])
# d = d.reshape([a,b,c])
# input_dict["fc1_expert_weights_for_ref"] = d


def f():
    gates = F.softmax(input_dict["gating_output"], axis=-1, dtype='float32') #.cast(input_dict["gating_output"].dtype)
    expert_scales, experts_for_row = paddle.topk(gates, k, axis=-1)
    denorm = paddle.clip(expert_scales.sum(-1, keepdim=True), min=1e-6)
    expert_scales = expert_scales / denorm  # [S,2]
    expert_scales = expert_scales.astype("float16")
    token_ids, token_weight, expert_ids_ptr = triton_moe_preposs(expert_scales, 
                                                             experts_for_row.astype("int32"), 
                                                             expert)
    M = input_dict["input_activations"].shape[0]
    K = input_dict["fc1_expert_weights_for_ref"].shape[1]
    N = input_dict["fc1_expert_weights_for_ref"].shape[2]
    N2 = K
    EM = token_ids.shape[0]

    from triton_ops import triton_moe
    intermediate_cache1 = paddle.zeros([M, k, N], dtype="float16")
    num_tokens_post_padded_ptr = paddle.to_tensor([-1], dtype="int32")
    triton_moe(
        input_dict["input_activations"].cast("float16"),
        input_dict["fc1_expert_weights_for_ref"],
        intermediate_cache1,
        token_weight,
        token_ids,
        expert_ids_ptr,
        num_tokens_post_padded_ptr, # num_tokens_post_padded
        N, # N
        K, # K
        EM, # EM
        (int)(experts_for_row.numel()), # num_valid_tokens
        K, # stride_am
        1, # stride_ak
        K * N, # stride_be
        N, # stride_bk
        1, # stride_bn
        N, # stride_cm
        1, # stride_cn
        1,
        0)
    
    x, gate = intermediate_cache1.chunk(2, axis=-1)
    tmp = F.silu(x) * gate
    intermediate_cache2 = paddle.zeros((M, k, N2), dtype="float16")

    N = N // 2

    from triton_ops import triton_moe2
    triton_moe2(
        tmp,
        input_dict["fc2_expert_weights_for_ref"],
        intermediate_cache2,
        expert_scales,
        token_ids,
        expert_ids_ptr,
        num_tokens_post_padded_ptr, # num_tokens_post_padded
        N2, # N
        N, # K
        EM, # EM
        (int)(experts_for_row.numel()), # num_valid_tokens
        N, # stride_am
        1, # stride_ak
        N * N2, # stride_be
        N2, # stride_bk
        1, # stride_bn
        N2, # stride_cm
        1, # stride_cn
        1,
        0)
    intermediate_cache2 = paddle.sum(intermediate_cache2, axis=1)
    return intermediate_cache2

import time
for i in range(10):
    start = time.perf_counter()
    ccc = f()
    paddle.device.cuda.synchronize()
    hf_cost = (time.perf_counter() - start) * 1000
    print("Speed triton:", hf_cost)


print(paddle.max(paddle.abs(ccc - bbb)))

