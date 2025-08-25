import paddle
from paddle import nn
import argparse
import os
import json
import paddle.profiler

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
        
def profile(args):
    # come distibuted envs
    paddle.distributed.init_parallel_env()
    rank = paddle.distributed.get_rank()
    local_rank = paddle.distributed.ParallelEnv().local_rank
    world_size = paddle.distributed.get_world_size()
    
    # set model
    model = nn.Linear(4096, 4096, bias_attr=False)
    model = model.to(f'gpu:{local_rank}')
    compute_tensor = paddle.randn((1024, 4096), dtype='float32')
    comm_tensor = paddle.randn((4096, 4096), dtype='float32')
    compute_tensor = compute_tensor.to(f'gpu:{local_rank}')
    comm_tensor = comm_tensor.to(f'gpu:{local_rank}')

    # set stream
    comm_stream = paddle.device.Stream()
    compute_stream = paddle.device.current_stream()
    paddle.device.synchronize()
    
    # set time list
    comm_time_list = []
    compute_time_list = []
    
    # define some functions
    def compute_func(dummy_input, iters):
        with paddle.device.stream_guard(compute_stream):
            for i in range(iters):
                output = model(compute_tensor)
                
    def comm_func(dummy_input, iters):
        with paddle.device.stream_guard(comm_stream):
            for i in range(iters):
                paddle.distributed.all_reduce(comm_tensor)
                
    def compute_comm_func(dummy_input, compute_iters, comm_iters):
        comm_func(dummy_input, comm_iters)
        compute_func(dummy_input, compute_iters)
        
    def analyse_json_file(log_path):
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
        
        local_comm_time_list, local_compute_time_list = [], []
        with open(log_path, "r") as f:
            data = json.load(f)
            traceEvents = data["traceEvents"] # traceEvents:List[Dict]
            for event in traceEvents:
                if "name" in event:
                    event_name = event["name"]
                    if "Kernel_AllReduce" in event_name: # when use develop branch, the name is ncclDevKernel_AllReduce, so we modify it to Kernel_AllReduce
                        start_time = timestr2timenum(event["args"]["start_time"])
                        end_time = timestr2timenum(event["args"]["end_time"])
                        local_comm_time_list.append(end_time - start_time)
                    elif "gemm" in event_name:
                        start_time = timestr2timenum(event["args"]["start_time"])
                        end_time = timestr2timenum(event["args"]["end_time"])
                        local_compute_time_list.append(end_time - start_time)
        
        if len(local_comm_time_list) != 0:
            ave_comm_time = sum(local_comm_time_list) / len(local_comm_time_list)
            ave_comm_time = paddle.to_tensor([ave_comm_time], dtype='float32', place=f"gpu:{local_rank}")
            paddle.distributed.all_reduce(ave_comm_time, op=paddle.distributed.ReduceOp.SUM)
            ave_comm_time = ave_comm_time.numpy()[0] / world_size
            comm_time_list.append(ave_comm_time)

        if  len(local_compute_time_list) != 0:
            ave_compute_time = sum(local_compute_time_list) / len(local_compute_time_list)
            ave_compute_time = paddle.to_tensor([ave_compute_time], dtype='float32', place=f"gpu:{local_rank}")
            paddle.distributed.all_reduce(ave_compute_time, op=paddle.distributed.ReduceOp.SUM)
            ave_compute_time = ave_compute_time.numpy()[0] / world_size
            compute_time_list.append(ave_compute_time)
        
    def profile_op(sync_stream, warmup_func, profile_func, op_type):
        with paddle.profiler.Profiler(
            targets=[paddle.profiler.ProfilerTarget.GPU],
            scheduler=(1, 2)
        ) as p:
            for i in range(2):
                dummy_input = None
                if i == 0:
                    printf("Warmup...")
                    warmup_func(dummy_input)
                else:
                    printf("Profile...")
                    profile_func(dummy_input)
                sync_stream.synchronize()
                p.step()
        
        log_path = f"./output/profile_overlap/profile_logs/rank{rank}_{op_type}.json"
        if not os.path.exists(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path))
        p.export(log_path) 
        analyse_json_file(log_path)
        
    # profile
    overlap_time_multiply = args.overlap_time_multiply
    
    # computation
    printf('Profiling computation time when not overlapped with communication...')
    profile_op(compute_stream, lambda x: compute_func(x, 512), lambda x: compute_func(x, 512), "compute")
    
    # communication
    printf('Profiling communication time when not overlapped with computation...')
    profile_op(comm_stream, lambda x: comm_func(x, 10), lambda x: comm_func(x, 30), "comm")
    
    # computation overlaps communication
    printf('\nProfiling communication time when overlapped with computation...')
    comm_iters = max(int(1000 / comm_time_list[0]), 5) # 1000 ms for communication
    compute_iters = int(overlap_time_multiply * comm_iters * comm_time_list[0] / compute_time_list[0])
    profile_op(comm_stream, lambda x: comm_func(x, comm_iters), lambda x: compute_comm_func(x, compute_iters, comm_iters), "compute_overlap_comm")
    comm_delay = comm_time_list[1] / comm_time_list[0]

    # communication overlaps computation
    printf('\nProfiling communication time when overlapped with computation...')
    compute_iters = max(int(1000 / compute_time_list[0]), 5) # 1000 ms for computation
    comm_iters = int(overlap_time_multiply * compute_iters * compute_time_list[0] / comm_time_list[0])
    profile_op(compute_stream, lambda x: comm_func(x, comm_iters), lambda x: compute_comm_func(x, compute_iters, comm_iters), "comm_overlap_compute")
    compute_delay = compute_time_list[2] / compute_time_list[0]

    # analyse data 
    overlap_coe = max(comm_delay, compute_delay)
    printf(f'Overlap coefficient: {overlap_coe}')
    printf(f'Comm delay: {comm_delay}')
    printf(f'Computation delay: {compute_delay}')
    printf(f'Comm time list: {comm_time_list} (unit: ms)')
    printf(f'Computation time list: {compute_time_list} (unit: ms)')
    
    # save the config
    save_file_name = "./configs/overlap_coefficient.json"
    save_config = {
        'overlap_coe': overlap_coe,
    }
    write_json_config(save_config, save_file_name)
    printf(f"Save overlap coefficient to {save_file_name}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Profiler")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--overlap_time_multiply', type=int, default=4, help='Overlap time multiply factor')
    args = parser.parse_args()
    profile(args)