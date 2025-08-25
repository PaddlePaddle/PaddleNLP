import numpy as np
import paddle.nn as nn
import paddle.distributed as dist
from paddle.io import BatchSampler, DataLoader, Dataset
import paddle
import paddle.profiler
from tqdm import tqdm
import os
import argparse
import json

def printf(string, prefix="[ProfileHardware] "):
    text_color = "\033[32m"
    reset_color = "\033[0m"
    print(f"{text_color}{prefix}{string}{reset_color}")

def read_json_config(path):
    if os.path.exists(path) == False:
        raise ValueError(f"Config file {path} not found.")
    return json.load(open(path, 'r', encoding="utf-8"))

def write_json_config(config, path):
    if os.path.exists(os.path.dirname(path)) == False:
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as fp:
        json.dump(config, fp, indent=4)

class RandomDataset(Dataset):
    def __init__(self, local_batch_size):
        self.dataset_size = local_batch_size * 11
        self.input = np.random.rand(*(self.dataset_size, 512, 1024))
        self.label = np.random.rand(*(self.dataset_size, 512, 1024))

    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError("Index out of range")
        input = paddle.to_tensor(self.input[idx], dtype='float32')
        label = paddle.to_tensor(self.label[idx], dtype='float32')
        return input, label

class LinearModel(nn.Layer):
    def __init__(self, pp_rank, pp_world_size, mesh):
        super().__init__()
        
        self.pp_rank = pp_rank
        self.pp_world_size = pp_world_size
        self.mesh = mesh
        
    def forward(self, x):
        x.stop_gradient = False
        for i in range(self.pp_world_size):
            if i != self.pp_world_size - 1:
                x = dist.reshard(x, self.mesh[i + 1], [dist.Replicate()])
        return x

class P2PCommModel(nn.Layer):
    def __init__(self, rank, send_rank, recv_rank, mesh):
        super().__init__()
        self.rank = rank
        self.send_rank = send_rank
        self.recv_rank = recv_rank
        self.mesh = mesh
        
    def forward(self, x):
        x.stop_gradient = False
        for i in range(2):
            if i == 0:
                x = dist.reshard(x, self.mesh[i + 1], [dist.Replicate()])
        return x
    
def test(args):
    dist.init_parallel_env()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = dist.ParallelEnv().local_rank
    group = dist.new_group(ranks=[i for i in range(world_size)])
    mesh = dist.ProcessMesh([i for i in range(world_size)], dim_names=['pp'])
    model = LinearModel(rank, world_size, mesh)
    
    local_batch_size = args.local_batch_size
    printf(f"local_batch_size: {local_batch_size}")
    dataset = RandomDataset(local_batch_size)
    sampler = BatchSampler(dataset, batch_size=local_batch_size)
    trainloader = DataLoader(dataset=dataset, batch_sampler=sampler)
    dist_dataloader = dist.shard_dataloader(trainloader, shard_dims=[0, 0], meshes=[mesh[0], mesh[-1]])
    
    # print some info
    p2p_message_size = local_batch_size * 512 * 1024 * 4 / 1024 / 1024
    printf(f"p2p_message_size: {p2p_message_size} MB")
    
    with paddle.profiler.Profiler(
        targets=[paddle.profiler.ProfilerTarget.GPU],
        scheduler=(1, 12),
        ) as p:
        for i, input in enumerate(tqdm(dist_dataloader)):
            data = input[0]
            out = model(data)
            p.step() 
            
    # save profiler result
    log_path = f'./output/profile_p2p/profile_logs/rank{rank}.json'
    if os.path.exists(os.path.dirname(log_path)) == False:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)    
    p.export(log_path)  
    
    # analyse profiler result
    def timestr2timenum(timestr):
        string = timestr.split(" ")
        num = float(string[0])
        unit = string[1]
        if unit == "ms":
            return num
        elif unit == "us":
            return num / 1000
        else:
            assert False, f"unit {unit} not supported"
    
    send_recv_time_list = []
    with open(log_path, "r") as f:
        data = json.load(f)
        traceEvents = data["traceEvents"]
        for event in traceEvents:
            if "name" in event:
                event_name = event['name']
                if "SendRecv" in event_name:
                    start_time = timestr2timenum(event["args"]["start_time"])
                    end_time = timestr2timenum(event["args"]["end_time"])
                    send_recv_time_list.append(end_time - start_time)
    
    send_recv_time_sum = sum(send_recv_time_list)
    send_recv_time_sum /= 10 # because we have 10 iterations in the dataloaders
    printf(f'send_recv_time_sum: {send_recv_time_sum} ms')
    
# def train(args):
#     dist.init_parallel_env()
#     world_size = dist.get_world_size()
#     rank = dist.get_rank()
    
#     send_rank = world_size // 2 - 1
#     recv_rank = world_size // 2
#     mesh = dist.ProcessMesh([send_rank, recv_rank], dim_names=['pp'])
#     printf(f'mesh is {mesh}')
    
#     # set model
#     if rank == send_rank or rank == recv_rank:
#         model = P2PCommModel(rank, send_rank, recv_rank, mesh)
#     else:
#         model = None
    
#     # set dataset
#     local_batch_size = args.local_batch_size
#     printf(f"local_batch_size: {local_batch_size}")
#     dataset = RandomDataset(local_batch_size)
#     sampler = BatchSampler(dataset, batch_size=local_batch_size)
#     trainloader = DataLoader(dataset=dataset, batch_sampler=sampler)
#     dist_dataloader = dist.shard_dataloader(trainloader, shard_dims=[0, 0], meshes=[mesh[0], mesh[-1]])
    
#     # print some info
#     p2p_message_size = local_batch_size * 512 * 1024 * 4 / 1024 / 1024
#     printf(f"p2p_message_size: {p2p_message_size} MB")
    
#     # profiler
#     with paddle.profiler.Profiler(
#         targets=[paddle.profiler.ProfilerTarget.GPU],
#         scheduler=(1, 12),
#         ) as p:
#         # for input in 
#         for i, input in enumerate(tqdm(dist_dataloader)):
#             data = input[0]
#             if rank == send_rank or rank == recv_rank:
#                 printf(f'step:{i}')
#                 out = model(data)
#             p.step()   
    
#     # save profiler result
#     save_file_path = f'./profile_logs/profile_p2p/rank{rank}.json'
#     if os.path.exists(os.path.dirname(save_file_path)) == False:
#         os.makedirs(os.path.dirname(save_file_path), exist_ok=True)    
#     p.export(save_file_path)
    
def get_groups(pp_deg):
    world_size = dist.get_world_size()
    total_ranks = list(range(0, world_size))
    assert world_size % pp_deg == 0
    
    group_size = world_size // pp_deg
    all_pp_groups = [total_ranks[i * group_size : (i + 1)  * group_size] for i in range(pp_deg)]
    send_groups = all_pp_groups[pp_deg // 2 - 1]
    recv_groups = all_pp_groups[pp_deg // 2]
    print(f'send_groups: {send_groups}')
    print(f'recv_groups: {recv_groups}')
    return send_groups, recv_groups

def use_dist_send_recv(args):
    dist.init_parallel_env()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = paddle.distributed.ParallelEnv().local_rank
    
    local_batch_size = args.local_batch_size
    printf(f"local_batch_size: {local_batch_size}")
    dataset = RandomDataset(local_batch_size)
    sampler = BatchSampler(dataset, batch_size=local_batch_size)
    trainloader = DataLoader(dataset=dataset, batch_sampler=sampler)
    
    send_groups, recv_groups = get_groups(args.pp_deg)
    send_groups_group = dist.new_group(ranks=send_groups)
    
    # print some info
    p2p_message_size = local_batch_size * 512 * 1024 * 4 / 1024 / 1024
    printf(f"p2p_message_size: {p2p_message_size} MB")
    
    with paddle.profiler.Profiler(
        targets=[paddle.profiler.ProfilerTarget.GPU],
        scheduler=(1, 12),
        ) as p:
        for i, input in enumerate(tqdm(trainloader)):
            data = input[0]
            data = data.to(paddle.get_device())
            if rank in send_groups:
                idx = send_groups.index(rank)
                dst_rank = recv_groups[idx]
                dist.send(data, dst=dst_rank)
                print(f'[linguangming] send_rank {rank}, dst_rank {dst_rank}')
            elif rank in recv_groups:
                idx = recv_groups.index(rank)
                src_rank = send_groups[idx]
                print(f'[linguangming] src_rank {src_rank}, recv_rank {rank}')
                dist.recv(data, src=src_rank)
            p.step() 
            
    # save profiler result
    log_path = f'./output/profile_p2p/profile_logs/rank{rank}.json'
    if os.path.exists(os.path.dirname(log_path)) == False:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)    
    p.export(log_path)  
    
    # analyse profiler result
    def timestr2timenum(timestr):
        string = timestr.split(" ")
        num = float(string[0])
        unit = string[1]
        if unit == "ms":
            return num
        elif unit == "us":
            return num / 1000
        else:
            assert False, f"unit {unit} not supported"
    
    if rank in send_groups:  
        send_recv_time_list = []
        with open(log_path, "r") as f:
            data = json.load(f)
            traceEvents = data["traceEvents"]
            for event in traceEvents:
                if "name" in event:
                    event_name = event['name']
                    if "SendRecv" in event_name:
                        start_time = timestr2timenum(event["args"]["start_time"])
                        end_time = timestr2timenum(event["args"]["end_time"])
                        send_recv_time_list.append(end_time - start_time)
                        
        send_recv_time_sum = sum(send_recv_time_list)
        send_recv_time_sum /= 10  # because we have 10 iterations in the dataloader
        comm_coe = p2p_message_size / send_recv_time_sum
        print(f'[linguangming] comm_coe is {comm_coe}')
        # if len(send_groups) != 1: 
        #     comm_coe = paddle.to_tensor([comm_coe], dtype='float32', place=f"gpu:{local_rank}")
        #     print(send_groups_group)
        #     paddle.distributed.all_reduce(comm_coe, group=send_groups_group, op=paddle.distributed.ReduceOp.SUM)
        #     comm_coe = comm_coe.numpy()[0] / send_groups_group.world_size
        
        printf(f'send_recv_time_sum: {send_recv_time_sum} ms')
        printf(f'comm_coe: {comm_coe} MB/ms')
        
        if rank == send_groups[0]:
            save_file_name = args.save_file_name
            if os.path.exists(os.path.dirname(save_file_name)) == False:
                os.makedirs(os.path.dirname(save_file_name), exist_ok=True)
            if os.path.exists(save_file_name) == False:
                with open(save_file_name, 'w') as f:
                    tmp = {}
                    json.dump(tmp, f, indent=4)
                    
            config = read_json_config(save_file_name)
            key = f'pp_size_{args.pp_deg}'
            config[key] = comm_coe
            write_json_config(config, save_file_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PaddlePaddle P2P Communication Profiler")
    parser.add_argument('--output_dir', type=str,)
    parser.add_argument("--local_batch_size", type=int, default=32, help="local batch size for each rank" )
    parser.add_argument("--save_file_name", type=str, default='./configs/', help="save file name")
    parser.add_argument('--pp_deg', type=int)
    args = parser.parse_args()

    use_dist_send_recv(args)