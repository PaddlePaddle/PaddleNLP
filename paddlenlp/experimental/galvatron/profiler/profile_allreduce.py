import paddle
from paddle import nn
from paddle.io import Dataset
import numpy as np
import random
from tqdm import tqdm
import argparse
import paddle.profiler
import json
import os
import paddle.distributed as dist

def read_json_config(path):
    if os.path.exists(path) == False:
        raise ValueError(f"Config file {path} not found.")
    return json.load(open(path, 'r', encoding="utf-8"))

def write_json_config(config, path):
    if os.path.exists(os.path.dirname(path)) == False:
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as fp:
        json.dump(config, fp, indent=4)

class pre_sync_module(nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, hidden_states):
        return hidden_states
    
class pre_mlp(nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1024, 1024)
        
    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        return hidden_states
    
def _reduce(input_, group):
    """all-reduce the input tensor across model parallel group"""
    if paddle.distributed.get_world_size(group=group) == 1:
        return input_
    paddle.distributed.all_reduce(input_.contiguous(), group=group)
    return input_

class _ReduceFromModelParallelRegion(paddle.autograd.PyLayer):
    """all-reduce ine input from the model parallel region"""
    
    @staticmethod
    def forward(ctx, input_, group):
        return _reduce(input_, group)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
class _CopyToModelParallelRegion(paddle.autograd.PyLayer):
    """pass the input to the model parallel region"""
    
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return input_
    
    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.group)
    
def reduce_from_model_parallel_region(input_, group):
    return _ReduceFromModelParallelRegion.apply(input_, group)

def copy_to_model_parallel_region(input_, group):
    return _CopyToModelParallelRegion.apply(input_, group)

class allreduce_block(nn.Layer):
    def __init__(self, mp_group):
        super().__init__()
        self.mp_group = mp_group
        
    def forward(self, hidden_states):
        hidden_states = copy_to_model_parallel_region(hidden_states, self.mp_group)
        hidden_states = reduce_from_model_parallel_region(hidden_states, self.mp_group)
        hidden_states.stop_gradient = False
        return hidden_states

class RandomDataset(Dataset):
    def __init__(self, local_batch_size, profile_time):
        self.dataset_size = local_batch_size * 11 # total 11 iteration
        self.input = np.random.rand(*(self.dataset_size, 512, 1024))
        self.profile_time = profile_time

    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError("Index out of range")
        if self.profile_time == 1:
            input = paddle.to_tensor(self.input[idx], dtype='bfloat16')
        else:
            input = paddle.to_tensor(self.input[idx], dtype='float32')
        return input
            
def fake_loss_func(labels, outputs):
    loss = outputs.sum()
    loss.stop_gradient = False
    return loss

def set_seed(rank):
    seed = 123 + rank
    np.random.seed(seed)
    random.seed(seed)
    paddle.seed(seed)

def get_tp_groups(tp_deg):
    world_size = dist.get_world_size()
    total_ranks = list(range(0, world_size))
    assert world_size % tp_deg == 0
    
    num = world_size // tp_deg
    all_tp_groups =  [total_ranks[i *  tp_deg : (i +  1) * tp_deg] for i in range(num)]
    all_tp_groups = [dist.new_group(ranks=tp_groups) for tp_groups in all_tp_groups]
    return all_tp_groups

def train(args):
    paddle.distributed.init_parallel_env()
    
    rank = paddle.distributed.get_rank()
    local_rank = paddle.distributed.ParallelEnv().local_rank
    set_seed(rank)
        
    args.num_layers = 24
    train_batch_size_input = args.local_batch_size
    print(f'local_batch_size: {train_batch_size_input}')

    dataset = RandomDataset(train_batch_size_input, args.profile_time)
    dataloader = paddle.io.DataLoader(dataset, batch_size=train_batch_size_input)    
    
    all_tp_groups = get_tp_groups(args.tp_deg)
    tp_group = None
    for group in all_tp_groups:
        if rank in group.ranks:
            tp_group = group
            break
    print(f"tp_group.nranks: {tp_group.nranks}")

    model = nn.Sequential()
    model.add_sublayer('pre_sync_module', pre_sync_module())
    model.add_sublayer('pre_mlp', pre_mlp())
    for i in range(args.num_layers):
        module = allreduce_block(tp_group)
        model.add_sublayer(f'mlp_{i}', module)

    if args.profile_time == 1:
        model = model.bfloat16()
        
    optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    
    # Calculate theoretical communication message size
    tp_deg = args.tp_deg
    local_batch_size = args.local_batch_size
    allreduce_message_size_per_layer = 2 * (tp_deg - 1) / tp_deg * (local_batch_size * 512 * 1024 * 2 * 4 / 1024 / 1024) # Multiply by 2 for bfloat16 size in bytes, multiply by 4 for 4 communications per forward-backward pass.
    allreduce_message_size_total = allreduce_message_size_per_layer * args.num_layers
    print(f"allreduce_message_size_per_layer: {allreduce_message_size_per_layer} MB")
    print(f"allreduce_message_size_total: {allreduce_message_size_total} MB")
    
    # train
    with paddle.profiler.Profiler(
        targets=[paddle.profiler.ProfilerTarget.GPU],
        scheduler=(1, 12),
        ) as p:
        # for input in 
        for i, input in enumerate(tqdm(dataloader)):
            input = input.to(f"gpu:{local_rank}")
            logits = model(input)
            loss = fake_loss_func(input, logits)
            loss.backward()
            optimizer.step
            optimizer.clear_grad()
            p.step()  
            
    log_path = f"./output/profile_allreduce/profile_logs/rank{rank}.json" 
    if os.path.exists(os.path.dirname(log_path)) == False:
        os.makedirs(os.path.dirname(log_path), exist_ok=True) 
    p.export(log_path)
    
    # analyse date
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
            
    comm_time = 0
    comm_num = 0
    with open(log_path) as f:
        data = json.load(f)
        traceEvents = data["traceEvents"]  # traceEvents:List[Dict]
        for event in traceEvents:
            if "name" in event:
                event_name = event["name"]
                if "AllReduce" in event_name:
                    comm_num += 1
                    start_time = timestr2timenum(event["args"]["start_time"])
                    end_time = timestr2timenum(event["args"]["end_time"])
                    comm_time += end_time - start_time
    print((f'comm_time: {comm_time} ms'))
    
    if args.profile_time == 0:  
        allreduce_time_24_layer = comm_time / 10 # [note] because we have 10 iterations in the dataloader
        comm_coe = allreduce_message_size_total / allreduce_time_24_layer
        comm_coe = paddle.to_tensor([comm_coe], dtype='float32', place=f"gpu:{local_rank}")
        paddle.distributed.all_reduce(comm_coe, group=tp_group, op=paddle.distributed.ReduceOp.SUM)
        comm_coe = comm_coe.numpy()[0] / tp_group.world_size
        print(f"comm_coe: {comm_coe} MB/ms")
        
        if rank == 0:
            save_file_name = args.save_file_name
            if os.path.exists(os.path.dirname(save_file_name)) == False:
                os.makedirs(os.path.dirname(save_file_name), exist_ok=True)
            if os.path.exists(save_file_name) == False:
                with open(save_file_name, 'w') as f:
                    tmp = {}
                    json.dump(tmp, f, indent=4)
            config = read_json_config(save_file_name)
            key = f'allreduce_size_{args.tp_deg}'
            config[key] = comm_coe
            write_json_config(config, save_file_name)
    else:
        per_comm_time = comm_time / comm_num
        per_comm_time = paddle.to_tensor([per_comm_time], dtype='float32', place=f"gpu:{local_rank}")
        dist.all_reduce(per_comm_time, group=tp_group, op=paddle.distributed.ReduceOp.SUM)
        comm_coe = per_comm_time.numpy()[0] / tp_group.world_size
        
        if rank == 0:
            save_file_name = args.save_file_name
            if os.path.exists(os.path.dirname(save_file_name)) == False:
                os.makedirs(os.path.dirname(save_file_name), exist_ok=True)
            if os.path.exists(save_file_name) == False:
                with open(save_file_name, 'w') as f:
                    tmp = {}
                    json.dump(tmp, f, indent=4)
            config = read_json_config(save_file_name)
            key = f'allreduce_size_{args.tp_deg}_{args.local_batch_size}MB_time'
            config[key] = comm_coe
            write_json_config(config, save_file_name)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Profiler allreduce")
    parser.add_argument('--output_dir', type=str,)
    parser.add_argument("--local_batch_size", type=int, default=64, help="local batch size")
    parser.add_argument("--profile_time", type=int, default=0, help="profile time")
    parser.add_argument("--save_file_name", type=str, default='./configs/', help="save file name")
    parser.add_argument('--tp_deg', type=int, default=1)
    
    args = parser.parse_args()
    
    train(args)