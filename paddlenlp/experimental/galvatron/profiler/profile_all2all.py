import paddle
import argparse
import paddle.distributed as dist
import numpy as np
import os
import json

def read_json_config(path):
    if os.path.exists(path) == False:
        raise ValueError(f"Config file {path} not found.")
    return json.load(open(path, 'r', encoding="utf-8"))

def write_json_config(config, path):
    if os.path.exists(os.path.dirname(path)) == False:
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as fp:
        json.dump(config, fp, indent=4)

def single_all_to_all(input, group):
    seq_world_size = dist.get_world_size(group)
    input_t = input.reshape((seq_world_size, -1))
    output = paddle.empty_like(input_t)
    dist.alltoall_single(output, input_t, group=group)
    return output

def get_tp_groups(tp_deg):
    world_size = dist.get_world_size()
    total_ranks = list(range(0, world_size))
    assert world_size % tp_deg == 0
    
    num = world_size // tp_deg
    all_tp_groups =  [total_ranks[i *  tp_deg : (i +  1) * tp_deg] for i in range(num)]
    all_tp_groups = [dist.new_group(ranks=tp_groups) for tp_groups in all_tp_groups]
    return all_tp_groups

def train(args):
    dist.init_parallel_env()
    rank  = dist.get_rank()
    local_rank = paddle.distributed.ParallelEnv().local_rank
    
    # get tp_groups
    all_tp_groups = get_tp_groups(args.tp_deg)
    tp_groups = None
    for groups in all_tp_groups:
        if rank in groups.ranks:
            tp_groups = groups
            break
    
    # print info
    local_batch_size = args.local_batch_size
    all2all_message_size_per_layer = local_batch_size * 512 * 1024 * 2 / 1024 /  1024
    if rank == 0:
        print(f'[all2all_messgae_size] per_layer {all2all_message_size_per_layer} MB')    
    
    # [Step1] warm up
    for _ in range(5):
        input = np.random.rand(*(local_batch_size, 512, 1024))
        input = paddle.to_tensor(input, dtype=paddle.bfloat16, place=f'gpu:{local_rank}')
        _ = single_all_to_all(input, tp_groups)
    
    # [Step2] real profile
    start = paddle.device.Event(enable_timing=True)
    end =  paddle.device.Event(enable_timing=True)
    time_list = []
    for _ in range(20):
        input = np.random.rand(*(local_batch_size, 512, 1024)) 
        input = paddle.to_tensor(input, dtype=paddle.bfloat16, place=f'gpu:{local_rank}')
        
        paddle.device.synchronize()
        dist.barrier(group=tp_groups) 
        start.record()
        _ = single_all_to_all(input, tp_groups)
        end.record()
        paddle.device.synchronize()
        
        duration = start.elapsed_time(end)
        print(f'device: {rank}, duration: {duration}')
        time_list.append(duration)
    
    # [Step3] store
    per_comm_time = sum(time_list) / len(time_list)
    per_comm_time = paddle.to_tensor([per_comm_time], dtype='float32', place=f'gpu:{local_rank}')
    dist.all_reduce(per_comm_time, group=tp_groups, op=dist.ReduceOp.SUM)
    per_comm_time = per_comm_time.numpy()[0] / tp_groups.world_size
    print(f'per_comm_time  is {per_comm_time}')
    
    # only store one tp groups result
    if rank == 0:
        save_file_name = args.save_file_name
        if os.path.exists(os.path.dirname(save_file_name)) == False:
            os.makedirs(os.path.dirname(save_file_name), exist_ok=True)
        if os.path.exists(save_file_name) == False:
            with open(save_file_name, 'w') as f:
                tmp = {}
                json.dump(tmp, f, indent=4)
        config = read_json_config(save_file_name)
        key = f'all2all_size_{args.tp_deg}_{args.local_batch_size}MB_time'
        config[key] = per_comm_time
        write_json_config(config, save_file_name)
        print('Already written all2all bandwidth into env config file %s!'%(save_file_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PaddlePaddle P2P Communication Profiler")
    parser.add_argument('--output_dirs', type=str,)
    parser.add_argument("--local_batch_size", type=int, default=32, help="local batch size for each rank" )
    parser.add_argument("--save_file_name", type=str, default='./configs/', help="save file name")
    parser.add_argument('--tp_deg', type=int, default=1,)
    args = parser.parse_args()
    train(args)