import paddle
from ..utils import read_json_config, write_json_config, num2str
import numpy as np
from paddle import core, framework
from dataclasses import dataclass, field
import os

@dataclass
class RuntimeProfilerArguments:
    # initialize from script
    profile_time_flag: int = field(default=0, metadata={"help": "Whether to enable time profiling."})
    profile_memory_flag: int = field(default=0, metadata={"help": "Whether to enable memory profiling."})
    profile_forward_only: int = field(default=0, metadata={"help": "Whether to enable forward profiling."})
    save_time_flag: int = field(default=0, metadata={"help": "Whether to save time profiling results."})
    save_memory_flag: int = field(default=0, metadata={"help": "Whether to save memory profiling results."})
    
    # initialize manually
    global_rank: int = field(default=0, metadata={"help": "The global rank of the process."})
    pp_degree: int = field(default=1, metadata={"help": "The degree of pipeline parallelism."})
    tp_degree: int = field(default=1, metadata={"help": "The degree of tensor parallelism."})
    dp_degree: int = field(default=1, metadata={"help": "The degree of data parallelism."})
    runtime_profiler_to_static: int = field(default=0, metadata={"help": "Whether to use static graph."})
    runtime_profiler_recompute: int = field(default=0, metadata={"help": "Whether to use runtime_profiler_recompute."})
    
    dp_rank: int = field(default=0, metadata={"help": "The rank of the data parallelism."})
    tp_rank: int = field(default=0, metadata={"help": "The rank of the tensor parallelism."})
    pp_rank: int = field(default=0, metadata={"help": "The rank of the pipeline parallelism."})
    
    model_name: str = field(default='llama', metadata={"help": "The model name for profiling."})
    layernum: int = field(default=16, metadata={"help": "The number of layers for profiling."})
    seq_len: int = field(default=1024, metadata={"help": "The sequence length for profiling."})
    mixed_precision: str = field(default='bf16', metadata={"help": "The mixed precision for profiling."})
    global_batch_size: int = field(default=8, metadata={"help": "The global batch size for profiling."})
    
class RuntimeProfiler:
    def __init__(self, args:RuntimeProfilerArguments):
        self.args = args
        
    # ================Time Profiling================
    def set_time_profiler(self, start_iter=10, end_iter=20): # [start_iter, end_iter)
        assert end_iter > start_iter, "End iteration must be greater than start iteration"
        
        self.start_iter = start_iter
        self.end_iter = end_iter
        
        self.start_event = paddle.device.Event(enable_timing=True)
        self.end_event = paddle.device.Event(enable_timing=True)
        self.time_list = []
    
    def profile_time_start(self, iter):
        if self.args.profile_time_flag == False:
            return
        
        if self.start_iter <= iter and iter < self.end_iter:
            paddle.device.synchronize()
            self.start_event.record()
        elif iter == self.end_iter:
            self._save_time_results()
        
    def profile_time_end(self, iter):
        if self.args.profile_time_flag == False:
            return
        
        if self.start_iter <= iter and iter < self.end_iter:
            self.end_event.record()
            paddle.device.synchronize()
            time_cost = self.start_event.elapsed_time(self.end_event)
            self.time_list.append(time_cost)
    
    def _save_time_results(self):
        if self.args.profile_time_flag == False:
            return
        
        ave_time = sum(self.time_list) / len(self.time_list)
        print(f"Average time cost: {ave_time:.4f} ms")
        print(f'original time cost: {self.time_list}')
        
        mean, std = np.mean(self.time_list), np.std(self.time_list)
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        print(f'Time mean: {mean:.4f} ms, std: {std:.4f} ms')
        print(f"Time cost range: [{lower_bound:.4f}, {upper_bound:.4f}]")
        self.time_list = [time for time in self.time_list if time >= lower_bound and time <= upper_bound]
        print(f"After removing outliers, time cost: {self.time_list}")
        
        if self.args.save_time_flag == 1:
            time_path = self.get_time_profiling_path()
            config = read_json_config(time_path) 
            
            layernum_info = num2str(self.args.layernum, "layernum")
            seq_info = num2str(self.args.seq_len, "seq")
            
            key = f'{layernum_info}_bsz{self.args.global_batch_size}_{seq_info}'
            config[key] = ave_time
            
            write_json_config(time_path, config)
            print(f"Already written profiled time into config file {time_path}!\n")
            
    # ================Memory Profiling================
    def set_memory_profiler(self, max_profile_memory_iter=5):
        self.max_profile_memory_iter = max_profile_memory_iter
        self.mem_dict = {}
        self.current_device = framework._current_expected_place_()
        print(f'[linguangming] pp_rank={self.args.pp_rank},  tp_rank={self.args.tp_rank},  dp_rank={self.args.dp_rank}')
        self.first_rank = True if self.args.pp_rank == 0 and self.args.tp_rank == 0 and self.args.dp_rank == 0 else False
        self.last_rank = True if self.args.pp_rank == self.args.pp_degree - 1 and self.args.tp_rank == self.args.tp_degree - 1 and self.args.dp_rank == self.args.dp_degree - 1 else False

    def profile_memory(self, iter, stage:str=""):
        if self.args.profile_memory_flag == False or iter > self.max_profile_memory_iter:
            return
        
        if stage == "Before Forward":
            core.device_memory_stat_reset_peak_value("Allocated", self.current_device.get_device_id())
        
        mem_dict = self.mem_dict
        max_memory_allocated = core.device_memory_stat_peak_value("Allocated", self.current_device.get_device_id()) / 2**20
        current_memory_allocated = core.device_memory_stat_current_value("Allocated", self.current_device.get_device_id()) / 2**20
        print(f'stage: {stage}, iter: {iter}, current_memory_allocated: {current_memory_allocated} MB, max_memory_allocated: {max_memory_allocated} MB')
        
        if stage == "Before Forward":
            mem_dict[f'iter_{iter}_before_forward'] = current_memory_allocated
        elif stage == "After Forward": # when use static graph, this stage is not used
            mem_dict[f'iter_{iter}_after_forward'] = current_memory_allocated
        elif stage == "After Backward":
            mem_dict[f'iter_{iter}_after_backward'] = current_memory_allocated
            mem_dict[f'iter_{iter}_after_backward_max'] = max_memory_allocated
        elif stage == "After Optimizer":
            pass

    def post_profile_memory(self, iter):
        if self.args.profile_memory_flag == False:
            return
        
        if iter == self.max_profile_memory_iter:
            """
                model_states: current memory allocated after backward
                
                model_states_and_peak_activation: max memory allocated after backward
                peak_activation: model_states_and_peak_activation - model_states
                
                model_states_and_activation: max memory allocated after forward
                activation: model_states_and_activation - current memory allocated before forward 
            """
            mem_dict = self.mem_dict
            mem_dict["model_states"] = mem_dict[f'iter_{self.max_profile_memory_iter - 1}_after_backward']
            mem_dict["model_states_and_peak_activation"] = mem_dict[f'iter_{self.max_profile_memory_iter - 1}_after_backward_max']
            mem_dict["peak_activation"] = mem_dict["model_states_and_peak_activation"] - mem_dict["model_states"]
            
            if self.args.runtime_profiler_to_static == 0: # only when use dynamic graph, the information of 'after forward' could get
                mem_dict['model_states_and_activation'] = mem_dict[f'iter_{self.max_profile_memory_iter - 1}_after_forward']
                mem_dict["activation"] = mem_dict[f'iter_{self.max_profile_memory_iter - 1}_after_forward'] - mem_dict[f"iter_{self.max_profile_memory_iter - 1}_before_forward"]

            print(f'Memory profiling results of rank[{self.args.global_rank}][pp_rank{self.args.pp_rank}][tp_rank{self.args.tp_rank}][dp_rank{self.args.dp_rank}]:')
            for key, val in mem_dict.items():
                print(f'\t{key}: {val:.2f} MB')
            
            if self.args.save_memory_flag != 0 and (self.first_rank or self.last_rank):
                assert self.args.runtime_profiler_to_static == 0, "Only support dynamic graph memory profiling"
                
                memory_path = self.get_memory_profiling_path()
                config = read_json_config(memory_path)
                
                args = self.args
                strategy_info = f'{args.pp_degree}_{args.tp_degree}_{args.dp_degree}_{args.runtime_profiler_recompute}'
                print(f'[linguangming] strategy_info is {strategy_info}')
                layernum_info = num2str(self.args.layernum, "layernum")
                seq_info = num2str(self.args.seq_len, "seq")

                if strategy_info not in config:
                    print(f'[linguangming] init empty')
                    config[strategy_info] = {}
                
                if self.first_rank:
                    print(f'[linguangming] this is first_rank')
                    config[strategy_info][f'{layernum_info}_bsz{args.global_batch_size}_{seq_info}_first_ms'] = mem_dict["model_states"]
                    config[strategy_info][f'{layernum_info}_bsz{args.global_batch_size}_{seq_info}_first_act'] = mem_dict["activation"]
                    config[strategy_info][f'{layernum_info}_bsz{args.global_batch_size}_{seq_info}_first_act_peak'] = mem_dict["peak_activation"]
                elif self.last_rank:
                    print(f'[linguangming] this is last_rank')
                    config[strategy_info][f'{layernum_info}_bsz{args.global_batch_size}_{seq_info}_last_ms'] = mem_dict["model_states"]
                    config[strategy_info][f'{layernum_info}_bsz{args.global_batch_size}_{seq_info}_last_act'] = mem_dict["activation"]
                    config[strategy_info][f'{layernum_info}_bsz{args.global_batch_size}_{seq_info}_last_act_peak'] = mem_dict["peak_activation"]
                
                
                print(f'[linguangming] config is {config}')
                write_json_config(memory_path, config)
                print(f"Already written profiled memory into config file {memory_path}!\n")
    
    # ================Utils Functions================
    def get_memory_profiling_path(self):
        assert self.first_rank or self.last_rank, "Only first rank and last rank can save memory profiling results"
        path = os.getcwd()
        if self.first_rank:
            memory_file_name = f'configs/memory_profiling_{self.args.mixed_precision}_{self.args.model_name}_first.json'
        elif self.last_rank:
            memory_file_name = f'configs/memory_profiling_{self.args.mixed_precision}_{self.args.model_name}_last.json'
        memory_path = os.path.join(path, memory_file_name)
        return memory_path
    
    def get_time_profiling_path(self):
        path = os.getcwd()
        time_file_name = f'configs/computation_profiling_{self.args.mixed_precision}_{self.args.model_name}_rank[{self.args.global_rank}].json'
        time_path = os.path.join(path, time_file_name)
        return time_path
        