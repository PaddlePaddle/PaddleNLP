from itertools import product
import sys
import copy
import os
from ..utils import read_json_config, write_json_config, num2str
from dataclasses import dataclass, field
from typing import List
from collections import defaultdict
import numpy as np

@dataclass
class ModelProfilerArguments:
    profile_type: str = field(default='memory', metadata={"help": "The type of profiling, either 'memory' or 'computation'."})
    profile_mode: str = field(default='static', metadata={"help": "The mode of profiling, either 'static', 'batch' or 'sequence'."})
    
    profile_fixed_batch_size: int = field(default=8, metadata={"help": "The fixed batch size for profiling."})
    profile_min_batch_size: int = field(default=1, metadata={"help": "The minimum batch size for profiling."})
    profile_max_batch_size: int = field(default=12, metadata={"help": "The maximum batch size for profiling."})
    profile_batch_size_step: int = field(default=1, metadata={"help": "The step size for batch size profiling."})
    
    layernum_min: int = field(default=2, metadata={"help": "The minimum number of layers for profiling."})
    layernum_max: int = field(default=4, metadata={"help": "The maximum number of layers for profiling."})
    
    profile_fixed_seq_length_list: str = field(default='1024,2048', metadata={"help": "The sequence length list for profiling."})
    profile_min_seq_length: int = field(default=1024, metadata={"help": "The minimum sequence length for profiling."})
    profile_max_seq_length: int = field(default=2048, metadata={"help": "The maximum sequence length for profiling."})
    profile_seq_length_step: int = field(default=1, metadata={"help": "The step size for sequence length profiling."})
    
    num_layertype: int = field(default=1, metadata={"help": "1:decoder-only and encoder-only, 2:encoder-decoder"})
    
    max_tp_deg: int = field(default=1, metadata={"help": "The maximum tensor parallel degree."})
    
    max_per_device_train_batch_size: int =  field(default=4, metadata={'help':''})

    def initialize(self, args_dict:dict):
        self.profile_type = args_dict.pop('--profile_type', 'memory')
        self.profile_mode = args_dict.pop('--profile_mode', 'static')
        self.profile_fixed_batch_size = int(args_dict.pop('--profile_fixed_batch_size', 1))
        self.profile_min_batch_size = int(args_dict.pop('--profile_min_batch_size', 1))
        self.profile_max_batch_size = int(args_dict.pop('--profile_max_batch_size', 12))
        self.profile_batch_size_step = int(args_dict.pop('--profile_batch_size_step', 1))
        self.layernum_min = int(args_dict.pop('--layernum_min', 2))
        self.layernum_max = int(args_dict.pop('--layernum_max', 4))
        self.profile_fixed_seq_length_list = args_dict.pop('--profile_fixed_seq_length_list', '1024,2048')
        self.num_layertype = int(args_dict.pop('--num_layertype', 1))
        self.max_tp_deg = int(args_dict.pop('--max_tp_deg', 1))
        self.profile_min_seq_length = int(args_dict.pop('--profile_min_seq_length', 1024))
        self.profile_max_seq_length = int(args_dict.pop('--profile_max_seq_length', 2048))
        self.profile_seq_length_step = int(args_dict.pop('--profile_seq_length_step', 1024))
        self.max_per_device_train_batch_size = int(args_dict.pop('--max_per_device_train_batch_size', 4))

class ModelProfiler:
    def __init__(self, args:ModelProfilerArguments, args_dict:dict):
        self.args = args
        self.args_dict = args_dict
        self.set_bsz_list()
        self.set_layernum_lists()
        self.set_seqlen_list()

    def launch_profiling(self):
        args = self.args
        if args.profile_type == 'memory':
            self.launch_memory_profiling_scripts()
        elif args.profile_type == 'computation':
            self.launch_time_profiling_scripts()
        else:
            raise ValueError(f"Unsupported profile type: {args.profile_type}. Supported types are 'memory' and 'computation'.")
    
    def process_data(self):
        args = self.args
        if args.profile_type == 'memory':
            self._process_memory_data()
        elif args.profile_type == 'computation':
            self._process_computation_data()
        else:
            raise ValueError(f"Unsupported profile type: {args.profile_type}. Supported types are 'memory' and 'computation'.")
    
    # =================Time Profiling================
    def launch_time_profiling_scripts(self):  # Currently only supports decoder-only architecture.
        CMD_LIST = []
        ARGS = copy.deepcopy(self.args_dict)
        for layernum_list in self.layernum_lists:  # self.layernum_lists = [[2, 2], [4, 2], [2, 4]] or [[2], [4]]
            for bsz in self.batch_size_list: # self.batch_size_list = [1, 2, 3, 4] or [4]
                for seq_tuple in self.product_sequence_length_list:  # self.product_sequence_length_list = [(1024, 1024), (1024, 2048), (2048, 1024), (2048, 2048)] or [(1024,), (2048,)]                    
                    ARGS['--profile_time_flag'] = 1
                    ARGS['--profile_forward_only'] = 1
                    
                    ARGS['--to_static'] = 0  # use dynamic graph
                    ARGS['--sharding_parallel_degree'] = 1
                    ARGS['--sharding'] = "stage2"  # when sharding_parallel_degree == 1, sharding set any value is ok
                    ARGS['--tensor_parallel_degree'] = 1
                    ARGS['--pipeline_parallel_degree'] = 1
                    
                    ARGS['--num_hidden_layers'] = layernum_list[0]  # Currently only supports decoder-only architecture.
                    ARGS['--seq_length'] = seq_tuple[0]
                    
                    ARGS['--per_device_train_batch_size'] = bsz // ARGS['--sharding_parallel_degree']
                    ARGS['--gradient_accumulation_steps'] = 1
                    
                    LAUNCHER = os.getenv('LAUNCHER')
                    
                    CMD = LAUNCHER + ' ' + ' '.join([f"{k} '{v}'" if isinstance(v, str) and ' ' in v else f"{k} {v}" for k, v in ARGS.items()])
                    CMD_LIST.append(CMD)

        print(f'[auto-parallel] All commands have been generated')
        for CMD in CMD_LIST:
            print(CMD)
                        
        print(f'[auto-parallel] Please run the following commands to get the time profiling data:')
        for CMD in CMD_LIST:
            print("[auto-parallel] run command: ", CMD)
            os.system(CMD)               
                    
    def _process_computation_data(self) -> None:
        time_config_path = self.get_time_profiling_path()
        config = read_json_config(time_config_path)
        
        for bsz in self.batch_size_list: # self.batch_size_list = [1, 2, 3, 4] or [4]
            for seq_tuple in self.product_sequence_length_list:  # self.product_sequence_length_list = [(1024, 1024), (1024, 2048), (2048, 1024), (2048, 2048)] or [(1024,), (2048,)]
                key_base = f'layernum[{self.args.layernum_min}]_bsz{bsz}_seq{seq_tuple[0]}'
                val_base = config[key_base]
                
                key = f'layernum[{self.args.layernum_max}]_bsz{bsz}_seq{seq_tuple[0]}'
                val = config[key]
                
                avg_time = (val - val_base) / bsz / (self.args.layernum_max - self.args.layernum_min)
                write_key = f"layertype_{0}_bsz{bsz}_seq{seq_tuple[0]}"
                config[write_key] = avg_time
                
                other_time = val_base
                other_time -= avg_time * bsz * self.args.layernum_min
                other_time /= bsz
                write_key = f"layertype_other_bsz{bsz}_seq{seq_tuple[0]}"
                config[write_key] = max(other_time, 0)
                
        write_json_config(time_config_path, config)
        print(f"Already written processed computation time into env config file {time_config_path}!\n")
   
    # =================Memory Profiling================
    def launch_memory_profiling_scripts(self):
        args = self.args
        assert args.profile_mode == 'static' or args.profile_mode == "sequence", 'memory profile support static and sequence mode'
        
        world_size = int(os.getenv('PROFILE_WORLD_SIZE'))
        max_tp_deg = min(world_size, args.max_tp_deg)
        if args.profile_mode != 'static':
            max_tp_deg = 1
        
        CMD_LIST = []
        ARGS = copy.deepcopy(self.args_dict)
        for seq_tuple in self.product_sequence_length_list:  # self.product_sequence_length_list = [(1024, 1024), (1024, 2048), (2048, 1024), (2048, 2048)] or [(1024,), (2048,)]
            pp_deg = 1
            for checkpoint in [0, 1]:
                tp_deg = 1
                while tp_deg <= max_tp_deg:
                    if pp_deg * tp_deg <= world_size:
                        for layernum_list in self.layernum_lists:
                            ARGS['--profile_memory_flag'] = 1
                            
                            ARGS['--to_static'] = 0  # use dynamic graph
                            ARGS['--pipeline_parallel_degree'] = pp_deg
                            ARGS['--tensor_parallel_degree'] = tp_deg
                            ARGS['--sharding_parallel_degree'] = world_size // pp_deg // tp_deg
                            ARGS['--sharding'] = "stage3"
                            
                            ARGS['--recompute'] = checkpoint
                            
                            ARGS['--num_hidden_layers'] = layernum_list[0]  # Currently only supports decoder-only architecture.
                            ARGS['--seq_length'] = seq_tuple[0]
                            
                            ARGS['--per_device_train_batch_size'] = args.profile_fixed_batch_size // ARGS['--sharding_parallel_degree']
                            ARGS['--per_device_train_batch_size'] = min(ARGS['--per_device_train_batch_size'], self.args.max_per_device_train_batch_size)
                            ARGS['--gradient_accumulation_steps'] = 1
                            
                            LAUNCHER = os.getenv('LAUNCHER')
                    
                            CMD = LAUNCHER + ' ' + ' '.join([f"{k} '{v}'" if isinstance(v, str) and ' ' in v else f"{k} {v}" for k, v in ARGS.items()])
                            CMD_LIST.append(CMD)
                            
                    if checkpoint:
                        break
                    tp_deg *= 2
            
            for pp_deg in [2, 4]:
                layer_num = pp_deg
                tp_deg = 1
                while tp_deg <= max_tp_deg:
                    if pp_deg * tp_deg <= world_size:
                        ARGS['--profile_memory_flag'] = 1

                        ARGS['--to_static'] = 0  # use dynamic graph
                        ARGS['--pipeline_parallel_degree'] = pp_deg
                        ARGS['--tensor_parallel_degree'] = tp_deg
                        ARGS['--sharding_parallel_degree'] = world_size // pp_deg // tp_deg
                        ARGS['--sharding'] = "stage3"
                        
                        ARGS['--recompute'] = 0
                        
                        ARGS['--num_hidden_layers'] = layer_num  # Currently only supports decoder-only architecture.
                        ARGS['--seq_length'] = seq_tuple[0]
                        
                        ARGS['--per_device_train_batch_size'] = args.profile_fixed_batch_size // ARGS['--sharding_parallel_degree']
                        ARGS['--per_device_train_batch_size'] = min(ARGS['--per_device_train_batch_size'], self.args.max_per_device_train_batch_size)
                        ARGS['--gradient_accumulation_steps'] = 1
                        
                        LAUNCHER = os.getenv('LAUNCHER')
                    
                        CMD = LAUNCHER + ' ' + ' '.join([f"{k} '{v}'" if isinstance(v, str) and ' ' in v else f"{k} {v}" for k, v in ARGS.items()])
                        CMD_LIST.append(CMD)
                    tp_deg *= 2
                    
        print(f'[auto-parallel] All commands have been generated')
        for CMD in CMD_LIST:
            print(CMD)
        
        print(f'[auto-parallel] Please run the following commands to get the memory profiling data:')
        for CMD in CMD_LIST:
            print("[auto-parallel] run command: ", CMD)
            os.system(CMD)
    
    def _process_memory_data(self):
        args = self.args
        
        # Merge the first and last rank memory profiling data
        first_rank_path, last_rank_path, file_path = self.get_memory_profiling_path()
        first_rank_config, last_rank_config = read_json_config(first_rank_path), read_json_config(last_rank_path)
        merged_data = {}
        for key in first_rank_config:
            merged_data[key] = {**first_rank_config[key], **last_rank_config[key]}
        write_json_config(file_path, merged_data)
        print(f'Merged memory profiling data has been written to: {file_path}')

        # process memory profiling data for each sequence length
        config = read_json_config(file_path)
        bsz = args.profile_fixed_batch_size # memory profiling only support static or sequence mode, and in this case, the batch size is fixed
        layernum_list_base = self.layernum_lists[0]
        layernum_lists_other = self.layernum_lists[1:]  # other layernum_list
        for seq_tuple in self.product_sequence_length_list:
            self._process_single_sequence_config(seq_tuple, config, layernum_list_base, layernum_lists_other, bsz)
    
        # Write the processed config back to the file
        write_json_config(file_path, config)
        print(f'Processed memory profiling data has been written to: {file_path}')

    def _process_single_sequence_config(self, seq_tuple, config, layernum_list_base:List[int], layernum_lists_other:List[List[int]], bsz:int):
        seq_info = num2str(list(seq_tuple), 'seq')
        print(f'Processing sequence length: {seq_tuple}')
        
        args = self.args
        # Initialize result containers
        param_result_list = [dict() for _ in range(args.num_layertype)]
        act_result_list = [dict() for _ in range(args.num_layertype)]
        param_list = [-1] * args.num_layertype
        
        # Get some information 
        world_size = int(os.getenv('PROFILE_WORLD_SIZE'))
        layernum_diff = args.layernum_max - args.layernum_min
        
        # [Step1] Process tensor paralleism memory costs (only use the case which pp_deg is 1 and recompute is False)
        fixed_pp_deg, tp_deg, fixed_recompute = 1, 1, False
        while fixed_pp_deg * tp_deg <= world_size:
            dp_deg = world_size // (fixed_pp_deg * tp_deg)
            strategy = f'{fixed_pp_deg}_{tp_deg}_{dp_deg}_{fixed_recompute}'
            if strategy in config: 
                re = config[strategy]
                for i in range(args.num_layertype):
                    layernum_key_0 = layernum_list_base 
                    layernum_key_1 = layernum_lists_other[i]
                    
                    bsz_adjust = self.adjust_bsz(gbsz=bsz, dp_deg=dp_deg)
                    # Calculate parameter memory per layer
                    model_states_divide_param = 9 # when use dynamic and O2 and zero3 and accumulation_steps is 1, model_states = param * 9
                    param_per_layer = (
                                        (re[self.key_format(layernum_key_1, bsz_adjust, seq_tuple[0], 'first', 'ms')] 
                                        - re[self.key_format(layernum_key_0, bsz_adjust, seq_tuple[0], 'first', 'ms')]) 
                                        / layernum_diff
                                        * fixed_pp_deg # this is unnessary
                                        / model_states_divide_param 
                                    )
                    param_per_layer *= dp_deg # when memory profile, we use zero-3. Now we restore the influence.
            
                    # Calculate activation memory per sample
                    act_per_layer_per_sample = (
                                                (re[self.key_format(layernum_key_1, bsz_adjust, seq_tuple[0], 'first', 'act')] 
                                                - re[self.key_format(layernum_key_0, bsz_adjust, seq_tuple[0], 'first', 'act')]) 
                                                / layernum_diff
                                            )
                    act_per_layer_per_sample *= dp_deg / bsz_adjust  # namely, act_per_layer_per_sample /= (bsz / dp_deg) 

                    # store the results
                    param_result_list[i][tp_deg] = param_per_layer
                    act_result_list[i][tp_deg] = act_per_layer_per_sample
                    param_list[i] = max(param_list[i], param_per_layer * tp_deg)
            tp_deg *= 2
            
        for i in range(args.num_layertype):
            print(f'layertype {i}:')
            print(f'param: {param_list[i]}')
            print(f'act_dict: {act_result_list[i]}')
            print(f'param_list: {param_result_list[i]}')

        # [Step2] Process checkpoint memory costs
        act_dict_c_list = [dict() for _ in range(args.num_layertype)]
        act_cpt_list = [-1] * args.num_layertype
        
        fixed_pp_deg, tp_deg, fixed_recompute = 1, 1, True
        while fixed_pp_deg * tp_deg <= world_size:
            dp_deg = world_size // (fixed_pp_deg * tp_deg)
            strategy = f'{fixed_pp_deg}_{tp_deg}_{dp_deg}_{fixed_recompute}'
            if strategy in config:
                re = config[strategy]
                for i in range(args.num_layertype):
                    layernum_key_0 = layernum_list_base
                    layernum_key_1 = layernum_lists_other[i]
                    
                    bsz_adjust = self.adjust_bsz(gbsz=bsz, dp_deg=dp_deg)
                    # Calculate activation memory with checkpointing
                    act_per_layer_per_sample = (
                                                (re[self.key_format(layernum_key_1, bsz_adjust, seq_tuple[0], 'first', 'act')]
                                                - re[self.key_format(layernum_key_0, bsz_adjust, seq_tuple[0], 'first', 'act')])
                                                / layernum_diff
                                                * tp_deg
                                            )
                    act_per_layer_per_sample *= dp_deg / bsz_adjust  # namely, act_per_layer_per_sample /= (bsz / dp_deg)
                    
                    act_per_layer_per_sample_max = (
                                                (re[self.key_format(layernum_key_1, bsz_adjust, seq_tuple[0], 'first', 'act_peak')]
                                                - re[self.key_format(layernum_key_0, bsz_adjust, seq_tuple[0], 'first', 'act_peak')])
                                                / layernum_diff
                                                * tp_deg
                                            )
                    act_per_layer_per_sample_max *= dp_deg / bsz_adjust  # namely, act_per_layer_per_sample /= (bsz / dp_deg)
                    
                    print(f'layertype {i} with checkpoint, tp_deg {tp_deg}: act_per_layer_per_sample = {act_per_layer_per_sample}')
                    act_dict_c_list[i][tp_deg] = max(act_per_layer_per_sample, act_per_layer_per_sample_max)
                    act_cpt_list[i] = max(act_cpt_list[i], act_per_layer_per_sample)
            tp_deg *= 2
        
        for i in range(args.num_layertype):
            print(f'layertype {i} with checkpoint:')
            print(f'act_cpt_dict: {act_dict_c_list[i]}')
            print(f'act_cpt: {act_cpt_list[i]}')
            act_result_list[i]['checkpoint'] = act_cpt_list[i]
        
        
        # [Step3] Process pipeline parallelism memory costs
        inf = 1e9
        other_memory_pp_off = {"model_states": defaultdict(lambda: inf), "activation": defaultdict(lambda: inf)}
        other_memory_pp_on_first = {"model_states": defaultdict(lambda: inf), "activation": defaultdict(lambda: inf)}
        other_memory_pp_on_last = {"model_states": defaultdict(lambda: inf), "activation": defaultdict(lambda: inf)}

        pp_deg, fixed_recompute = 1, False
        while pp_deg <= world_size:
            tp_deg = 1
            while pp_deg * tp_deg <= world_size:
                dp_deg = world_size // (pp_deg * tp_deg)
                strategy = f'{pp_deg}_{tp_deg}_{dp_deg}_{fixed_recompute}'

                if strategy in config:
                    re = config[strategy]

                    layernum = pp_deg if pp_deg > 1 else layernum_list_base[0]
                    layernum_list = [layernum] * args.num_layertype
                    
                    model_states_divide_param = 9  # when use dynamic and O2 and zero3 and accumulation_steps is 1, model_states = param * 9
                    ms_cost = [param_result_list[l][tp_deg] * model_states_divide_param for l in range(args.num_layertype)]
                    act_cost = [act_result_list[l][tp_deg] for l in range(args.num_layertype)]

                    # Calculate total memory costs for first and last pipeline stages
                    layer_ms_costs_first = self.total_memcost(pp_deg, layernum, args.num_layertype, ms_cost, 0)
                    layer_ms_costs_last = self.total_memcost(pp_deg, layernum, args.num_layertype, ms_cost, pp_deg - 1)
                    layer_act_costs_first = self.total_memcost(pp_deg, layernum, args.num_layertype, act_cost, 0)
                    layer_act_costs_last = self.total_memcost(pp_deg, layernum, args.num_layertype, act_cost, pp_deg - 1)
                    
                    # Calculate other memory costs (Actually, this is unused)
                    # other_ms_first = re[self.key_format(layernum_list, bsz, seq_tuple[0], 'first', "ms")] - layer_ms_costs_first
                    # other_ms_last = re[self.key_format(layernum_list, bsz, seq_tuple[0], 'last', "ms")] - layer_ms_costs_last
                    
                    # Adjust for ZeRO-3 (default use zero3)
                    bsz_adjust = self.adjust_bsz(gbsz=bsz, dp_deg=dp_deg)
                    other_ms_first = (re[self.key_format(layernum_list, bsz_adjust, seq_tuple[0], 'first', "ms")] - layer_ms_costs_first / dp_deg) * dp_deg
                    other_ms_last = (re[self.key_format(layernum_list, bsz_adjust, seq_tuple[0], 'last', "ms")] - layer_ms_costs_last / dp_deg) * dp_deg

                    # Calculate activation memory peaks
                    if pp_deg != 1:
                        act_peak_first = max(re[self.key_format(layernum_list, bsz_adjust, seq_tuple[0], 'first', "act_peak")], re[self.key_format(layernum_list, bsz_adjust, seq_tuple[0], 'first', "act")])
                        act_peak_last = max(re[self.key_format(layernum_list, bsz_adjust, seq_tuple[0], 'last', "act_peak")], re[self.key_format(layernum_list, bsz_adjust, seq_tuple[0], 'last', "act")])
                    else:
                        # 这个地方好像写错了
                        act_peak_first = re[self.key_format(layernum_list, bsz_adjust, seq_tuple[0], 'first', "act")]
                        act_peak_last = re[self.key_format(layernum_list, bsz_adjust, seq_tuple[0], 'last', "act")]
                        # act_peak_first = max(re[self.key_format(layernum_list, bsz_adjust, seq_tuple[0], 'first', "act_peak")], re[self.key_format(layernum_list, bsz_adjust, seq_tuple[0], 'first', "act")])
                        # act_peak_last = max(re[self.key_format(layernum_list, bsz_adjust, seq_tuple[0], 'last', "act_peak")], re[self.key_format(layernum_list, bsz_adjust, seq_tuple[0], 'last', "act")])
                        
                    other_act_first = (act_peak_first - layer_act_costs_first * bsz_adjust / dp_deg) / (bsz_adjust / dp_deg)
                    other_act_last = (act_peak_last - layer_act_costs_last * bsz_adjust / dp_deg) / (bsz_adjust / dp_deg)
                    
                    # Ensure non-negative memory costs
                    other_ms_first = max(other_ms_first, 0)
                    other_ms_last = max(other_ms_last, 0)
                    other_act_first = max(other_act_first, 0)
                    other_act_last = max(other_act_last, 0)
                    
                    # Store the results
                    tp_key = tp_deg # NOTE Need to modify when fine-grained support is added
                    if pp_deg == 1:
                        other_memory_pp_off["model_states"][tp_key] = min(other_memory_pp_off["model_states"][tp_key], other_ms_first)
                        other_memory_pp_off["activation"][tp_key] = min(other_memory_pp_off["activation"][tp_key], other_act_first)
                        # other_memory_pp_off["model_states"][tp_key] = max(other_memory_pp_off["model_states"][tp_key], other_ms_first)
                        # other_memory_pp_off["activation"][tp_key] = max(other_memory_pp_off["activation"][tp_key], other_act_first)
                    else:
                        # other_memory_pp_on_first["model_states"][tp_key] = max(other_memory_pp_on_first["model_states"][tp_key], other_ms_first)
                        # other_memory_pp_on_first["activation"][tp_key] = max(other_memory_pp_on_first["activation"][tp_key], other_act_first)
                        # other_memory_pp_on_last["model_states"][tp_key] = max(other_memory_pp_on_last["model_states"][tp_key], other_ms_last)
                        # other_memory_pp_on_last["activation"][tp_key] = max(other_memory_pp_on_last["activation"][tp_key], other_act_last)
                        other_memory_pp_on_first["model_states"][tp_key] = min(other_memory_pp_on_first["model_states"][tp_key], other_ms_first)
                        other_memory_pp_on_first["activation"][tp_key] = min(other_memory_pp_on_first["activation"][tp_key], other_act_first)
                        other_memory_pp_on_last["model_states"][tp_key] = min(other_memory_pp_on_last["model_states"][tp_key], other_ms_last)
                        other_memory_pp_on_last["activation"][tp_key] = min(other_memory_pp_on_last["activation"][tp_key], other_act_last)
                tp_deg *= 2
            pp_deg *= 2
        print('other_memory_pp_off:', other_memory_pp_off)
        print('other_memory_pp_on_first:', other_memory_pp_on_first)
        print('other_memory_pp_on_last:', other_memory_pp_on_last)
        
        # [Step4] Adjust some tp_deg
        for tp_deg in [2, 4, 8]:
            for i in range(args.num_layertype):
                if tp_deg not in act_result_list[i]:
                    act_result_list[i][tp_deg] = act_result_list[i][tp_deg // 2] / 2
            for memory_dict in [other_memory_pp_off, other_memory_pp_on_first, other_memory_pp_on_last]:
                for key in ['model_states', 'activation']:
                    if tp_deg not in memory_dict[key]:
                        memory_dict[key][tp_deg] = memory_dict[key][tp_deg // 2] / 2
        print('After adjust tp_deg 2, 4, 8:')
        print('act_result_list:', act_result_list)
        print('other_memory_pp_off:', other_memory_pp_off)
        print('other_memory_pp_on_first:', other_memory_pp_on_first)
        print('other_memory_pp_on_last:', other_memory_pp_on_last)
    
        # [Step5] Write the results into config files
        for i in range(args.num_layertype):
            config_key = f'layertype_{i}'
            if config_key not in config:
                config[config_key] = {}
            config[config_key][seq_tuple[i]] = {
                'parameter_size' : param_list[i],
                'tp_activation_per_bsz_dict': act_result_list[i],
            }
            
        memory_keys = {
            "other_memory_pp_off": other_memory_pp_off,
            "other_memory_pp_on_first": other_memory_pp_on_first,
            "other_memory_pp_on_last": other_memory_pp_on_last,
        }
        for config_key, value in memory_keys.items():
            if config_key not in config:
                config[config_key] = {}
            config[config_key][seq_info[3:]] = copy.deepcopy(value)
            
    # =================Utils Functions================
    def set_bsz_list(self):
        args = self.args
        if args.profile_mode == 'static':
            assert args.profile_fixed_batch_size is not None
            self.batch_size_list = [args.profile_fixed_batch_size]
        elif args.profile_mode == 'batch':
            assert args.profile_min_batch_size is not None and args.profile_max_batch_size is not None and args.profile_batch_size_step is not None, 'please set the min batch size, max batch size and batch size step'
            self.batch_size_list = list(range(args.profile_min_batch_size, args.profile_max_batch_size + 1, args.profile_batch_size_step))
        elif args.profile_mode == 'sequence':
            self.batch_size_list = [args.profile_fixed_batch_size]

        print(f'[auto-parallel] batch size list: {self.batch_size_list}')
    
    def set_layernum_lists(self):
        args = self.args
        self.layernum_lists = []
        
        base_list = [args.layernum_min] * args.num_layertype
        self.layernum_lists.append(base_list)
        
        for idx in range(args.num_layertype):
            lst = base_list.copy()
            lst[idx] = args.layernum_max
            self.layernum_lists.append(lst)
        
        print(f'[auto-parallel] layernum_lists: {self.layernum_lists}')        
    
    def set_seqlen_list(self): 
        self.sequence_length_list = []
        
        args = self.args
        if args.profile_mode == 'static' or args.profile_mode == 'batch':
            assert args.profile_fixed_seq_length_list is not None, 'please set the seq length list'
            profile_seq_length_list = list(map(int, args.profile_fixed_seq_length_list.split(',')))
            for i in range(args.num_layertype):
                self.sequence_length_list.append([profile_seq_length_list[i]])
        elif args.profile_mode == 'sequence':
            assert args.num_layertype == 1, 'sequence length profiling only support one layer type'
            assert args.profile_min_seq_length is not None and args.profile_max_seq_length is not None and args.profile_seq_length_step is not None, 'please set the min seq length, max seq length and seq length step'
            
            if args.profile_type == 'memory':
                assert args.profile_min_seq_length is not None and args.profile_max_seq_length is not None
                # For memory profiling, sequence lengths must be powers of 2
                assert (
                    1 << (args.profile_min_seq_length.bit_length() - 1)
                ) == args.profile_min_seq_length, "profile_min_seq_length must be a power of 2"
                assert (
                    1 << (args.profile_max_seq_length.bit_length() - 1)
                ) == args.profile_max_seq_length, "profile_max_seq_length must be a power of 2"
                self.sequence_length_list.append(
                    [
                        (1 << j)
                        for j in range(
                            args.profile_min_seq_length.bit_length() - 1, args.profile_max_seq_length.bit_length()
                        )
                    ]
                )
            elif args.profile_type == 'computation':
                for i in range(args.num_layertype):
                    self.sequence_length_list.append(list(range(args.profile_min_seq_length, args.profile_max_seq_length + 1, args.profile_seq_length_step)))

        self.product_sequence_length_list = list(product(*self.sequence_length_list))
        
        print(f'[auto-parallel] sequence length list: {self.sequence_length_list}')            
        print(f'[auto-parallel] product sequence length list: {self.product_sequence_length_list}')

    def get_memory_profiling_path(self):
        path = os.getcwd()
        model_name = self.args_dict['--model_name_or_path']
        mixed_precision = 'bf16' if self.args_dict.get('--bf16') else 'fp16' if self.args_dict.get('--fp16') else 'fp32'
        first_rank_file_name = f'configs/memory_profiling_{mixed_precision}_{model_name}_first.json'
        last_rank_file_name = f'configs/memory_profiling_{mixed_precision}_{model_name}_last.json'
        file_name = f'configs/memory_profiling_{mixed_precision}_{model_name}.json'
        first_rank_path = os.path.join(path, first_rank_file_name)
        last_rank_path = os.path.join(path, last_rank_file_name)
        file_path = os.path.join(path, file_name)
        return first_rank_path, last_rank_path, file_path
    
    def get_time_profiling_path(self):
        path = os.getcwd()
        model_name = self.args_dict['--model_name_or_path']
        mixed_precision = 'bf16' if self.args_dict.get('--bf16') else 'fp16' if self.args_dict.get('--fp16') else 'fp32'
        time_file_name = f'configs/computation_profiling_{mixed_precision}_{model_name}_rank[0].json'  # when time profiling, only one gpu is used
        time_path = os.path.join(path, time_file_name)
        return time_path
    
    def key_format(self, layernum:List[int], bsz:int, seq:int, rank:str, type:str):
        text = ""
        text += f'layernum[{",".join(map(str, layernum))}]'
        text += f'_bsz{bsz}'
        text += f'_seq{seq}'
        text += f'_{rank}' # 'first' or 'last'
        text += f'_{type}' # 'ms' or 'act' or 'act_peak'
        return text
    
    def total_memcost(self, pp_deg, layernum, layertype, per_layer_cost:List[float], stage_idx:int):
        layer_costs = []
        for i in range(layertype):
            layer_costs.extend([per_layer_cost[i]] * layernum)
        
        total_layer_num = layernum * layertype
        avg_layer_num = int(total_layer_num // pp_deg)
        last_layer_num = total_layer_num - avg_layer_num * (pp_deg - 1)
        pp_divide = [avg_layer_num] * (pp_deg - 1) + [last_layer_num]
        
        # Verify equal distribution
        assert avg_layer_num == last_layer_num
        
        start_idx = int(np.sum(pp_divide[:stage_idx]))
        end_idx = int(np.sum(pp_divide[: stage_idx + 1]))
        return np.sum(layer_costs[start_idx:end_idx])
        
    def adjust_bsz(self, gbsz, dp_deg):
        fixed_accumulation_steps = 1
        per_device_train_batch_size = gbsz // fixed_accumulation_steps // dp_deg
        if per_device_train_batch_size <= self.args.max_per_device_train_batch_size:
            return gbsz
        else:
            return self.args.max_per_device_train_batch_size * dp_deg * fixed_accumulation_steps