import paddle
import numpy as np
from paddle.distributed import fleet
import paddle.distributed as dist


from paddle.incubate.nn.functional import (
    moe_dispatch,
    moe_ffn,
    moe_reduce,
    swiglu,
)

total_cards = 16
attention_tp = 1

strategy = fleet.DistributedStrategy()
strategy.hybrid_configs = {"dp_degree": total_cards // attention_tp, "mp_degree": attention_tp, "pp_degree": 1}
fleet.init(is_collective=True, strategy=strategy)

hcg = fleet.get_hybrid_communicate_group()

mp_group = hcg.get_model_parallel_group()

dtype = "bfloat16"
hidden_size = 7168
moe_intermediate_size = 2048
experts_num = 256
top_k = 8

ep_num_per_gpu = experts_num // total_cards

# 所有的卡的gate_weights是相同的！
gate_weights = paddle.randn([hidden_size, experts_num])
dist.broadcast(gate_weights, src=0)

ffn1_weights = paddle.randn([ep_num_per_gpu, hidden_size, moe_intermediate_size * 2], dtype="bfloat16")
ffn2_weights = paddle.randn([ep_num_per_gpu, moe_intermediate_size, hidden_size], dtype="bfloat16")
ffn1_biases = None
ffn2_biases = None
ffn1_weights_scale =  None
ffn2_weights_scale =  None
quant_type = "None"
norm_topk_prob=False

if ep_num_per_gpu == 1:
    one_ffn1_weights = ffn1_weights[0]
    one_ffn2_weights = ffn2_weights[0]
else:
    one_ffn1_weights = None
    one_ffn2_weights = None



USE_NVSHMEM = False
if USE_NVSHMEM:
    import pynvshmem
    rank = paddle.distributed.get_rank()
    world_size = dist.get_world_size()
    device = None
    nvshmem = pynvshmem.NvshmemCommunicator(rank, world_size, device)


group_num = total_cards // attention_tp
group_id = paddle.distributed.get_rank() // attention_tp
first_mp_id_in_group = group_id * attention_tp
IsFirstGPUInAttentionTP = paddle.distributed.get_rank() == first_mp_id_in_group


def MoE_FFN(permute_input, token_nums_per_expert,  all_ffn1_weights, all_ffn2_weights):
    if True:
        ffn_out = moe_ffn(
            permute_input,
            token_nums_per_expert,
            all_ffn1_weights,
            all_ffn2_weights,
            None,
            None,
            None,
            "None",
        )
    else:
        local_ep_num = all_ffn1_weights.shape[0]
        for i in range(local_ep_num):
            weight_A = all_ffn1_weights[i]
            weight_B = all_ffn2_weights[i]
            start = 0
            if i > 0:
                start = token_nums_per_expert[i-1]
            end = token_nums_per_expert[i]
            x = permute_input[start:end]

            tmp_out1 = paddle.matmul(x, weight_A)
            tmp_out1 = swiglu(tmp_out1)
            ffn_out = paddle.matmul(tmp_out1, weight_B)
            permute_input[start:end] = ffn_out
        ffn_out = paddle.assign(permute_input)
    return ffn_out



def compute_baseline(tmp_out):
    all_tensor = []
    dist.all_gather(all_tensor, ffn1_weights)
    all_ffn1_weights = paddle.concat(all_tensor, axis=0)
    all_tensor = []
    dist.all_gather(all_tensor, ffn2_weights)
    all_ffn2_weights = paddle.concat(all_tensor, axis=0)

    gate_out = paddle.matmul(tmp_out.cast("float32"), gate_weights)

    (
        permute_input,
        token_nums_per_expert,
        permute_indices_per_token,
        expert_scales_float,
        top_k_indices,
    ) = moe_dispatch(tmp_out, gate_out, top_k, False)

    ffn_out = MoE_FFN(permute_input, token_nums_per_expert, all_ffn1_weights, all_ffn2_weights)

    if USE_NVSHMEM:
        expert_scales_float[:,:] = 0.0
        expert_scales_float[:,0] = 1.0

    fused_moe_out = moe_reduce(
        ffn_out,
        expert_scales_float,
        permute_indices_per_token,
        top_k_indices,
        ffn2_biases,
        norm_topk_prob,
    )
    return fused_moe_out

def get_adjacent_minus(x):
    y = paddle.assign(x)
    y[1:] = y[0:-1]
    y[0] = 0
    y = x - y
    return y

start_event = paddle.device.Event(enable_timing=True)
end_event = paddle.device.Event(enable_timing=True)
alltoall_before_ffn = []
ffn_time = []
alltoall_after_ffn = []

def compute_ep_moe(tmp_out):
    assert ep_num_per_gpu == 1
    dtype = tmp_out.dtype
    
    start_event.record()

    gate_out = paddle.matmul(tmp_out.cast("float32"), gate_weights)

    end_event.record()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"gate_compute: {round(elapsed_time_ms,2)} ms")

    start_event.record()

    (
        permute_input,
        token_nums_per_expert,
        permute_indices_per_token,
        expert_scales_float,
        top_k_indices,
    ) = moe_dispatch(tmp_out, gate_out, top_k, False)


    end_event.record()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"moe_dispatch: {round(elapsed_time_ms,2)} ms")


    
    paddle.device.synchronize()
    start_event.record()
    
    if IsFirstGPUInAttentionTP:
        act_in_split_size = get_adjacent_minus(token_nums_per_expert)
        permute_input = permute_input
    else:
        act_in_split_size = paddle.zeros([total_cards], token_nums_per_expert.dtype)
        permute_input = paddle.empty([0, hidden_size], dtype=dtype)
    
    act_out_split_size = paddle.empty_like(act_in_split_size)
    dist.alltoall(act_out_split_size, act_in_split_size)

    this_card_token_nums = act_out_split_size.sum().reshape([1])

    act_in_split_size = act_in_split_size.numpy().tolist()
    act_out_split_size = act_out_split_size.numpy().tolist()

    permute_input_per_card = paddle.empty([this_card_token_nums, hidden_size], dtype=dtype)
    dist.alltoall_single(permute_input_per_card, permute_input, act_in_split_size, act_out_split_size)

    end_event.record()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"2alltoall: {round(elapsed_time_ms,2)} ms")

    start_event.record()

    tmp_out1 = paddle.matmul(permute_input_per_card, one_ffn1_weights)
    tmp_out1 = swiglu(tmp_out1)
    ffn_out = paddle.matmul(tmp_out1, one_ffn2_weights)

    # ffn_out = moe_ffn(permute_input_per_card, this_card_token_nums, ffn1_weights, ffn2_weights, ffn1_biases, ffn1_weights_scale, ffn2_weights_scale, quant_type)
    
    end_event.record()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"ffn: {round(elapsed_time_ms,2)} ms")

    start_event.record()
    dist.alltoall_single(permute_input, ffn_out, act_out_split_size, act_in_split_size,sync_op=True)
    moe_reduce_input = paddle.assign(permute_input)

    end_event.record()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"1alltoall: {round(elapsed_time_ms,2)} ms")

    start_event.record()

    fused_moe_out = moe_reduce(
        moe_reduce_input,
        expert_scales_float,
        permute_indices_per_token,
        top_k_indices,
        ffn2_biases,
        norm_topk_prob,
    )

    end_event.record()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"moe_reduce: {round(elapsed_time_ms,2)} ms")

    return fused_moe_out


def compute_moe_ep_with_tp_dp(tmp_out):
    act_dtype = tmp_out.dtype
    IsFirstGPUInAttentionTP = fleet.get_hybrid_communicate_group().get_model_parallel_rank() == 0

    gate_out = paddle.matmul(tmp_out.cast("float32"), gate_weights)

    (
        permute_input,
        token_cumsum_by_expert,
        permute_indices_per_token,
        expert_scales_float,
        top_k_indices,
    ) = moe_dispatch(tmp_out, gate_out, top_k, False)



    if IsFirstGPUInAttentionTP:
        permute_input = permute_input
        token_num_by_expert = get_adjacent_minus(token_cumsum_by_expert).reshape([total_cards, ep_num_per_gpu])
        act_in_split_size = token_num_by_expert.sum(axis=-1)
    else:
        # Give a fake input, because we do not need get activation from this gpu.
        permute_input = paddle.empty([0, hidden_size], act_dtype)
        token_num_by_expert = paddle.zeros_like(token_cumsum_by_expert).reshape([total_cards, ep_num_per_gpu])
        act_in_split_size = paddle.zeros([total_cards], token_cumsum_by_expert.dtype)

    # allocate space for token_num_from_all_cards [total_cards, ep_num_per_gpu]
    token_num_from_all_cards = paddle.empty_like(token_num_by_expert)

    
    paddle.device.synchronize()
    start_event.record()

    dist.alltoall(token_num_from_all_cards, token_num_by_expert)
    
    # allocate space for permute_input_per_card
    act_out_split_size = token_num_from_all_cards.sum(axis=-1)
    permute_input_per_card = paddle.empty([act_out_split_size.sum(), hidden_size], act_dtype)
    act_in_split_size = act_in_split_size.numpy().tolist()
    act_out_split_size = act_out_split_size.numpy().tolist()
    
    # permute_input_per_card = permute_input_per_card.cast("int8")
    # permute_input = permute_input.cast("int8")

    dist.alltoall_single(permute_input_per_card, permute_input, act_in_split_size, act_out_split_size)

    # permute_input_per_card = permute_input_per_card.cast("bfloat16")
    # permute_input = permute_input.cast("bfloat16")

    paddle.device.synchronize()
    end_event.record()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    alltoall_before_ffn.append(elapsed_time_ms)
    print(f"2alltoall: {round(elapsed_time_ms,2)} ms")

    def run_permute_input(act_after_all2all, flag=True):
    
        if ep_num_per_gpu == 1:
            # not need reorder.
            return act_after_all2all
        result = paddle.empty_like(act_after_all2all)
        
        # token_num_from_all_cards [total_cards, ep_num_per_gpu]
        # compute in cpu.
        tmp = token_num_from_all_cards.numpy().reshape(-1)
        tmp = tmp.cumsum() - tmp
        token_cumsum_from_all_cards = tmp.tolist()

        token_num_from_all_cards_list = token_num_from_all_cards.numpy().tolist()

        # index_select_indices = [0] * act_out_split_size.sum().item()
        index_select_indices = [0] * sum(act_out_split_size)
        j = 0
        for i in range(ep_num_per_gpu):
            for in_gpu_id in range(total_cards):
                num = token_num_from_all_cards_list[in_gpu_id][i]
                j1 = token_cumsum_from_all_cards[in_gpu_id * ep_num_per_gpu + i]
                if num > 0:
                    if flag:
                        index_select_indices[j:j+num] = list(range(j1,j1+num))
                    else:
                        index_select_indices[j1:j1+num] = list(range(j,j+num))
                j += num
        if len(index_select_indices) > 0:
            result = act_after_all2all.index_select(paddle.to_tensor(index_select_indices))
        return result
    
    permute_input_per_card = run_permute_input(permute_input_per_card)
    
    token_cumsum_by_expert_per_card = token_num_from_all_cards.transpose([1,0]).sum(axis=-1).cumsum()
    
    # ffn_out = moe_ffn(permute_input_per_card, token_cumsum_by_expert_per_card, ffn1_weights, ffn2_weights, ffn1_biases, ffn1_weights_scale, ffn2_weights_scale, quant_type)

    paddle.device.synchronize()
    start_event.record()

    ffn_out = MoE_FFN(permute_input_per_card, token_cumsum_by_expert_per_card, ffn1_weights, ffn2_weights)

    paddle.device.synchronize()
    end_event.record()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    ffn_time.append(elapsed_time_ms)
    print(f"ffn: {round(elapsed_time_ms,2)} ms")

    ffn_out = run_permute_input(ffn_out, False)

    paddle.device.synchronize()
    start_event.record()

    dist.alltoall_single(permute_input, ffn_out, act_out_split_size, act_in_split_size)
    moe_reduce_input = permute_input

    paddle.device.synchronize()
    end_event.record()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    alltoall_after_ffn.append(elapsed_time_ms)
    print(f"1alltoall: {round(elapsed_time_ms,2)} ms")

    fused_moe_out = moe_reduce(
        moe_reduce_input,
        expert_scales_float,
        permute_indices_per_token,
        top_k_indices,
        ffn2_biases,
        norm_topk_prob,
    )

    return fused_moe_out





def compute_ep_moe_nvshmeme(tmp_out):
    assert ep_num_per_gpu == 1

    start_event.record()

    gate_out = paddle.matmul(tmp_out.cast("float32"), gate_weights)

    end_event.record()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"gate_compute: {round(elapsed_time_ms,2)} ms")

    start_event.record()

    (
        permute_input,
        token_nums_per_expert,
        permute_indices_per_token,
        expert_scales_float,
        top_k_indices,
    ) = moe_dispatch(tmp_out, gate_out, top_k, False)


    end_event.record()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"moe_dispatch: {round(elapsed_time_ms,2)} ms")
    
    start_event.record()
    
    permute_input_per_card, handle_ptr = nvshmem.all_to_all_dispatch(
        tmp_out,
        top_k_indices,
    )

    end_event.record()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"2alltoall: {round(elapsed_time_ms,2)} ms")

    start_event.record()

    tmp_out1 = paddle.matmul(permute_input_per_card, one_ffn1_weights)
    tmp_out1 = swiglu(tmp_out1)
    ffn_out = paddle.matmul(tmp_out1, one_ffn2_weights)
    
    end_event.record()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"ffn: {round(elapsed_time_ms,2)} ms")

    start_event.record()
    haha = nvshmem.all_to_all_combine(
        ffn_out,
        top_k_indices,
        handle_ptr
    )
    haha = haha.reshape([-1, top_k, hidden_size])
    expert_scales_float[:,:] = 0.0
    expert_scales_float[:,0] = 1.0
    haha = haha * expert_scales_float.unsqueeze(-1)
    haha = haha.sum(axis=1)

    end_event.record()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"1alltoall: {round(elapsed_time_ms,2)} ms")
    return haha

if __name__ == '__main__':
    
    from paddle.framework import core

    for bs in [64, 128, 256]:
        core.nvprof_start()
        for i in range(20):
            flush_cache = paddle.randn([512, 512, 512], dtype)

            tmp_x = paddle.randn([bs, hidden_size], dtype)
            tmp_x = tmp_x * 0.01
            if USE_NVSHMEM:
                res = compute_ep_moe_nvshmeme(tmp_x)
            else:
                # res = compute_ep_moe(tmp_x)
                res = compute_moe_ep_with_tp_dp(tmp_x)
        core.nvprof_stop()

        baseline = compute_baseline(tmp_x)

        print("batch batch size", bs)
        
        print(round(np.percentile(alltoall_before_ffn, 80), 2))
        print(round(np.percentile(ffn_time, 80), 2))
        print(round(np.percentile(alltoall_after_ffn, 80), 2))

        alltoall_before_ffn = []
        ffn_time = []
        alltoall_after_ffn = []

        paddle.distributed.barrier()
        if res.shape[0] > 0:
            print(res.shape)
            print(baseline.shape)
            print((res-baseline).abs().max())


# data = paddle.ones([256, hidden_size])

# dist.all_reduce(data, group=mp_group)

# print(data)
# exit(0)

# all_tensor = []
# dist.all_gather(all_tensor, data)
# dist.all_gather(all_tensor, data)

# start_event.record()

# dist.all_gather(all_tensor, data)

# end_event.record()
# elapsed_time_ms = start_event.elapsed_time(end_event)
# print(f"dist.all_gather: {round(elapsed_time_ms,2)} ms")

# print(data)




