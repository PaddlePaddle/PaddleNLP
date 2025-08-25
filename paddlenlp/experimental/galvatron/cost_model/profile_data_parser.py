from dataclasses import dataclass, field
from scipy.optimize import curve_fit
from ..utils import num2str, read_json_config, Strategy, LayerWiseStrategy
from .memory_cost_model import MemoryCostModel, MemoryCostModelArguments, OtherMemoryCostModelArguments, OtherMemoryCostModel
from .time_cost_model import TimeCostModel, TimeCostModelArguments, OtherTimeCostModel, OtherTimeCostModelArguments
from typing import List
import numpy as np

@dataclass
class ProfileDataParserArguments:
    time_profile_mode: str = field(default='static', metadata={"help": "The mode of time profiling."})
    time_profile_data_path: str = field(default=None, metadata={"help": "The path of time profiling data."})
    
    memory_profile_mode: str = field(default='static', metadata={"help": "The mode of memory profiling."})
    memory_profile_data_path: str = field(default=None, metadata={"help": "The path of memory profiling data."})

    overlap_coe_path: str = field(default=None, metadata={"help": "The path of overlap coefficient data."})
    allreduce_coe_path: str = field(default=None, metadata={"help": "The path of allreduce coefficient data."})
    p2p_coe_path: str = field(default=None, metadata={"help": "The path of p2p coefficient data."})

    num_layertype: int = field(default=1, metadata={"help": "1: decoder-only or encoder-only, 2: encoder-decoder"})
    hidden_size_list: str = field(default="4096", metadata={"help": "The hidden size of the model."})
    layernum_list: str = field(default="12", metadata={"help": "The number of layers of the model."})
    seqlen_list: str = field(default="1024", metadata={"help": "The sequence length of the model."})
    
    profile_gpu_num: int = field(default=8, metadata={"help": "The number of GPUs used for profiling."})
    
    sp_time_path: str = field(default=None, metadata={"help": "The path of the sp time data."})
    
    def initialize(self, args_dict):
        self.time_profile_mode = args_dict.pop('--time_profile_mode', self.time_profile_mode)
        self.time_profile_data_path = args_dict.pop('--time_profile_data_path', self.time_profile_data_path)
        self.memory_profile_mode = args_dict.pop('--memory_profile_mode', self.memory_profile_mode)
        self.memory_profile_data_path = args_dict.pop('--memory_profile_data_path', self.memory_profile_data_path)
        self.overlap_coe_path = args_dict.pop('--overlap_coe_path', self.overlap_coe_path)
        self.allreduce_coe_path = args_dict.pop('--allreduce_coe_path', self.allreduce_coe_path)
        self.p2p_coe_path = args_dict.pop('--p2p_coe_path', self.p2p_coe_path)
        self.num_layertype = int(args_dict.pop('--num_layertype', self.num_layertype))
        self.hidden_size_list = args_dict.pop('--hidden_size_list', self.hidden_size_list)
        self.layernum_list = args_dict.pop('--layernum_list', self.layernum_list)
        self.seqlen_list = args_dict.pop('--seqlen_list', self.seqlen_list)
        self.profile_gpu_num = int(args_dict.pop('--profile_gpu_num', self.profile_gpu_num))
        self.sp_time_path = args_dict.pop('--sp_time_path', self.sp_time_path)

    
class ProfileDataParser:
    """
        A class to parse the profile data from json file.
    """
    def __init__(self, args:ProfileDataParserArguments):
        self.args = args
        self.validate_args()
        self.parse_profile_computation_configs()
        self.parse_profile_memory_configs()
        self.parse_profile_hardware_configs()
        
    def validate_args(self):
        args = self.args
        assert args.time_profile_mode in ['static', 'batch', 'sequence'], f'Unsupported time profile mode: {args.time_profile_mode}'
        assert args.memory_profile_mode in ['static', 'sequence'], f'Unsupported memory profile mode: {args.memory_profile_mode}'
        assert args.time_profile_data_path is not None, f'Time profile data path is None'
        assert args.memory_profile_data_path is not None, f'Memory profile data path is None'
        
        self.hidden_size_list = [int(x) for x in args.hidden_size_list.split(',')]
        self.layernum_list = [int(x) for x in args.layernum_list.split(',')]
        self.seqlen_list = [int(x) for x in args.seqlen_list.split(',')]
        assert len(self.hidden_size_list) == args.num_layertype and len(self.layernum_list) == args.num_layertype and len(self.seqlen_list) == args.num_layertype, f'The length of hidden_size_list, layernum_list and seqlen_list should be equal to num_layertype: {args.num_layertype}, but got {len(self.hidden_size_list)}, {len(self.layernum_list)}, {len(self.seqlen_list)}'
    
    # ==================Parse Profile Data=================
    def parse_profile_computation_configs(self):
        print("Parsing profile computation configs...")
        args = self.args
        self.time_profiled_list = []  # transformer layers
        self.other_time_profiled_list = []  # embedding layer, classifier layer, etc. Actuall, other_time is independent of num_layertype, but we still use a List to store it.
        self.time_config = read_json_config(args.time_profile_data_path)
        
        if args.time_profile_mode == 'static':
            for i in range(args.num_layertype):
                for key, value in self.time_config.items(): # the format of key is like layertype_0_bsz8_seq1024, layertype_other_bsz8_seq1024
                    if key.startswith(f'layertype_{i}_'):
                        self.time_profiled_list.append(value)
                    elif key.startswith(f'layertype_other_'):
                        self.other_time_profiled_list.append(value)
        elif args.time_profile_mode == 'batch':
            # process transformer layers
            for i in range(args.num_layertype): 
                x_data, y_data = [], []  
                for key, value in self.time_config.items():  # the format of key is like layertype_0_bsz8_seq1024, layertype_other_bsz8_seq1024
                    if key.startswith(f'layertype_{i}_') and f'_seq{self.seqlen_list[i]}' in key:
                        bsz = int(key.split('_')[-2][3:])  
                        x_data.append(bsz)
                        y_data.append(value * bsz)
                assert len(x_data) >= 8, f'Different batch size data is less than 8, please check the time profile data'
                # fit using a linear function
                def linear_func(x, m, c):
                    return m * x + c
                popt, _ = curve_fit(linear_func, x_data, y_data)
                self.time_profiled_list.append(popt)
                print("\tFitted popt for transformer layers:", popt)
            # process other layers, like embedding layer, classifier layer, etc.
            for i in range(args.num_layertype):
                x_data, y_data = [], []  
                for key, value in self.time_config.items(): # the format of key is like layertype_0_bsz8_seq1024, layertype_other_bsz8_seq1024
                    if key.startswith(f'layertype_other_') and f'_seq{self.seqlen_list[i]}' in key:
                        bsz = int(key.split('_')[-2][3:])  
                        x_data.append(bsz)
                        y_data.append(value * bsz)
                assert len(x_data) >= 8, f'Different batch size data is less than 8, please check the time profile data'
                # fit using a linear function
                def linear_func(x, m, c):
                    return m * x + c
                popt, _ = curve_fit(linear_func, x_data, y_data)
                self.other_time_profiled_list.append(popt)
                print("\tFitted popt for other layers:", popt)
        elif args.time_profile_mode == 'sequence':
            # process transformer layers
            for i in range(args.num_layertype):
                x_data, y_data = [], []  
                for key, value in self.time_config.items():
                    if key.startswith(f'layertype_{i}_') and f'_bsz1_' in key:
                        x_data.append(int(key.split('seq')[-1]))
                        y_data.append(value)
                # fit using a quadratic function
                def quadratic_func(x, a, b, c):
                    return a * x ** 2 + b * x + c
                popt, _ = curve_fit(quadratic_func, x_data, y_data)
                print("\tFitted popt for transformer layers:", popt)
                self.time_profiled_list.append(quadratic_func(self.seqlen_list[i], *popt))
                print(f'\tforward_time is {quadratic_func(self.seqlen_list[i], *popt)}')
            
            # process other layers, like embedding layer, classifier layer, etc.
            for i in range(args.num_layertype):
                x_data, y_data = [], []  
                for key, value in self.time_config.items():
                    if key.startswith(f'layertype_other_') and f'_bsz1_' in key:
                        x_data.append(int(key.split('seq')[-1]))
                        y_data.append(value)
                # fit using a linear function
                def linear_func(x, m, c):
                    return m * x + c
                popt, _ = curve_fit(linear_func, x_data, y_data)
                print("\tFitted popt for other layers:", popt)
                self.other_time_profiled_list.append(linear_func(self.seqlen_list[i], *popt))
                print(f'\tother forward_time is {linear_func(self.seqlen_list[i], *popt)}')
                
        else:
            raise ValueError(f"Unsupported time profile mode: {args.time_profile_mode}")
        print("Profile computation configs parsed successfully.")
    
    def parse_profile_memory_configs(self):
        print("Parsing profile memory configs...")
        args = self.args
        self.memory_config = read_json_config(args.memory_profile_data_path)
        self.memory_config = self.strkeys2intkeys(self.memory_config)
        
        self.param_sizes = [0 for _ in range(args.num_layertype)]
        self.act_sizes = [{} for _ in range(args.num_layertype)]
        
        if args.memory_profile_mode == 'sequence':
            assert args.num_layertype == 1, f'num_layertype should be 1 when memory profile mode is sequence, but got {args.num_layertype}'
            maxseq_list = []
            for i in range(args.num_layertype):
                layer_mem_config = self.memory_config[f'layertype_{i}']
                seqs = layer_mem_config.keys()
                maxseq = max([int(seq) for seq in seqs])
                minseq = min([int(seq) for seq in seqs])
                maxseq_list.append(maxseq)
                parameter_size = layer_mem_config[minseq]['parameter_size']
                tp_activation_per_bsz_dict = layer_mem_config[maxseq]['tp_activation_per_bsz_dict'].copy()
                self.param_sizes[i] = parameter_size
                self.act_sizes[i] = tp_activation_per_bsz_dict
                # adjust activation
                for tp in self.act_sizes[i]:
                    self.act_sizes[i][tp] = self.act_sizes[i][tp] / maxseq * self.seqlen_list[i]
            self.other_memory_pp_off = self.memory_config['other_memory_pp_off'][maxseq_list[0]]
            self.other_memory_pp_on = {'first_stage':self.memory_config['other_memory_pp_on_first'][maxseq_list[0]], 'last_stage':self.memory_config['other_memory_pp_on_last'][maxseq_list[-1]]}
            # adjust activation
            for tp in self.other_memory_pp_off['activation']:
                self.other_memory_pp_off['activation'][tp] = self.other_memory_pp_off['activation'][tp] / maxseq_list[0] * self.seqlen_list[0] 
                self.other_memory_pp_on['first_stage']['activation'][tp] = self.other_memory_pp_on['first_stage']['activation'][tp] / maxseq_list[0] * self.seqlen_list[0] 
                self.other_memory_pp_on['last_stage']['activation'][tp] = self.other_memory_pp_on['last_stage']['activation'][tp] / maxseq_list[-1] * self.seqlen_list[-1] 
        elif args.memory_profile_mode == 'static':
            for i in range(args.num_layertype):
                layer_mem_config = self.memory_config[f'layertype_{i}']
                parameter_size = layer_mem_config[self.seqlen_list[i]]['parameter_size']
                tp_activation_per_bsz_dict = layer_mem_config[self.seqlen_list[i]]['tp_activation_per_bsz_dict'].copy()
                self.param_sizes[i] = parameter_size
                self.act_sizes[i] = tp_activation_per_bsz_dict
            seq_info = num2str(self.seqlen_list, 'seq')[3:]
            if seq_info.isdigit():
                seq_info = int(seq_info)
            self.other_memory_pp_off = self.memory_config['other_memory_pp_off'][int(seq_info)]  # NOTE 此处貌似并没有管不同num_layertype下的其他序列信息
            self.other_memory_pp_on = {'first_stage':self.memory_config['other_memory_pp_on_first'][seq_info], 'last_stage':self.memory_config['other_memory_pp_on_last'][seq_info]}
            print(f'\tparameter sizes: {self.param_sizes}')
            print(f'\tactivation sizes: {self.act_sizes}')
            print(f'\tother memory pp off: {self.other_memory_pp_off}')
            print(f'\tother memory pp on: {self.other_memory_pp_on}')
        else:
            raise ValueError(f"Unsupported memory profile mode: {args.memory_profile_mode}")
        print("Profile memory configs parsed successfully.")
        
    def parse_profile_hardware_configs(self):
        print("Parsing profile hardware configs...")
        args = self.args
        
        # parse overlap coefficient
        self.overlap_coe = read_json_config(args.overlap_coe_path)['overlap_coe']
        print(f'\tOverlap coefficient: {self.overlap_coe}')
        
        # parse allreduce coefficient
        config = read_json_config(args.allreduce_coe_path)
        self.allreduce_coe = {}
        allreduce_size = args.profile_gpu_num
        while allreduce_size >= 2:
            if f'allreduce_size_{allreduce_size}' in config.keys():
                self.allreduce_coe[allreduce_size] = 1 / config[f'allreduce_size_{allreduce_size}']
            allreduce_size //= 2
        self.allreduce_coe[1] = 0
        print(f'\tAllreduce coefficient: {self.allreduce_coe}')
        
        # parse p2p coefficient
        config = read_json_config(args.p2p_coe_path)
        self.p2p_coe = {}
        for key, value in config.items():
            if 'pp_size' in key:
                pp_size = int(key.split('_')[-1])
                self.p2p_coe[pp_size] = 1 / value
        print(f'\tP2P coefficient: {self.p2p_coe}')        
        
        # parse sp time data
        config = read_json_config(args.sp_time_path)
        def remap_config(config, op):
            remap_config = {}
            for key, val in config.items():
                if key.startswith(op):
                    if op == "allreduce":
                        val /= 2 # trans to all_gather / reduce_scatter time
                    split = key.split("_")
                    world_size, size = int(split[-3]), int(split[-2][:-2])
                    if world_size in remap_config:
                        remap_config[world_size][size * 1024 * 1024] = val
                    else:
                        remap_config[world_size] = {}
                        remap_config[world_size][size * 1024 * 1024] = val
            
            for world_size, time_config in remap_config.items():
                x_data = []
                y_data = []
                for size, time in time_config.items():
                    x_data.append(size // 1024 // 1024)
                    y_data.append(time)
                
                # 临时注释 因为64和32的tp并行度还是不完整的
                # assert len(x_data) >= 8, f"Different size in communication profile of {op} should not be lower than 8."
            
                def linear_func(x, m, c):
                    return m * x + c
                popt, pcov = curve_fit(linear_func, x_data, y_data)
                
                print(f"Fitted parameters of {op}", popt)
                
                time_config["popt"] = popt
                
            return remap_config
        self.sp_allreduce = remap_config(config, "allreduce")
        self.sp_all2all = remap_config(config, "all2all")
        print(f'\tSP allreduce time: {self.sp_allreduce}')
        print(f'\tSP p2p time: {self.sp_all2all}')
        
        print("Profile hardware configs parsed successfully.")
            
    # ==================Launch Cost Model=================
    def get_memory_cost_for_specific_strategy(self, strategy:LayerWiseStrategy, global_batch_size:int, mixed_precision_type:str, accumulation_steps:int) -> List[float]:
        args = self.args
        
        # get some basic information
        pp_size = strategy.pp_size
        world_size = strategy.pp_size * strategy.tp_size * strategy.dp_size
        sharding_stage = strategy.sharding_stage
        
        # Memory consumption across different layers at various pipeline stages
        memory_per_layer_each_stage = [dict() for _ in range(args.num_layertype)]  
        for i in range(args.num_layertype):
            for stage_idx in range(pp_size):
                memory_cost_model_args = MemoryCostModelArguments(strategy=strategy, global_batch_size=global_batch_size, mixed_precision_type=mixed_precision_type, stage_idx=stage_idx, accumulation_steps=accumulation_steps, parameter_memory=self.param_sizes[i], tp_activation_per_bsz_dict=self.act_sizes[i])
                re = MemoryCostModel(memory_cost_model_args).get_memory_cost()
                memory_per_layer_each_stage[i][stage_idx] = re['enc_total']
        print(f'\tMemory cost for each layer type at each stage: {memory_per_layer_each_stage}')
        
        # Calculate other layer memory costs
        other_memory_cost_model_args = OtherMemoryCostModelArguments(min_tp_size=strategy.tp_size, max_tp_size=strategy.tp_size, world_size=world_size, pp_size=pp_size, sharding_stage=sharding_stage, global_batch_size=global_batch_size, accumulation_steps=accumulation_steps, other_memory_pp_off=self.other_memory_pp_off, other_memory_pp_on=self.other_memory_pp_on)
        memory_other = OtherMemoryCostModel(other_memory_cost_model_args).get_other_memory_cost()
        print(f'\tMemory cost for other layers: {memory_other}')
        
        # if use recompute, calculate not use recompute
        if strategy.recompute != 0:
            import copy
            strategy_no_recompute = copy.deepcopy(strategy)
            strategy_no_recompute.recompute = 0
            print("strategy_no_recompute is ", strategy_no_recompute)
            memory_per_layer_each_stage_no_recompute = [dict() for _ in range(args.num_layertype)] 
            for i in range(args.num_layertype):
                for stage_idx in range(pp_size):
                    mbsz = global_batch_size // accumulation_steps // strategy.dp_size 
                    add_activation = self.act_sizes[i][strategy.tp_size] * mbsz
                    memory_per_layer_each_stage_no_recompute[i][stage_idx] = add_activation
                    # memory_cost_model_args = MemoryCostModelArguments(strategy=strategy_no_recompute, global_batch_size=global_batch_size, mixed_precision_type=mixed_precision_type, stage_idx=stage_idx, accumulation_steps=accumulation_steps, parameter_memory=self.param_sizes[i], tp_activation_per_bsz_dict=self.act_sizes[i])
                    # re = MemoryCostModel(memory_cost_model_args).get_memory_cost()
                    # memory_per_layer_each_stage_no_recompute[i][stage_idx] = re['enc_total']
            print(f'\tMemory cost for each layer type at each stage (no recompute): {memory_per_layer_each_stage_no_recompute}')

        # compose the memory cost of each stage
        if pp_size == 1:
            memory_cost_per_stage = [0]
            memory_cost_per_stage[0] += memory_other[strategy.tp_size][0]
            for i in range(args.num_layertype):
                memory_cost_per_stage[0] += memory_per_layer_each_stage[i][0] * self.layernum_list[i]
            for i in range(args.num_layertype):
                if strategy.recompute != 0:
                    memory_cost_per_stage[0] += memory_per_layer_each_stage_no_recompute[i][0]
        else:
            memory_cost_per_stage = [0 for _ in range(pp_size)]
            memory_cost_per_stage[0] += memory_other[strategy.tp_size][0]
            memory_cost_per_stage[-1] += memory_other[strategy.tp_size][-1]
            
            total_layer_num = sum(self.layernum_list)
            avg_layer_num = total_layer_num // pp_size
            last_stage_layer_num = total_layer_num - avg_layer_num * (pp_size - 1)
            layers = []
            for i in range(args.num_layertype):
                layer_num = self.layernum_list[i]
                layers.extend([i] * layer_num)
            stage_layers = [layers[i * avg_layer_num: (i + 1) * avg_layer_num] for i in range(pp_size - 1)] + [layers[-last_stage_layer_num:]]
            for stage_idx in range(pp_size):
                for layer_type in stage_layers[stage_idx]:
                    memory_cost_per_stage[stage_idx] += memory_per_layer_each_stage[layer_type][stage_idx]
            if strategy.recompute != 0:
                for stage_idx in range(pp_size):
                    if stage_idx == pp_size - 1:
                        continue
                    memory_cost_per_stage[stage_idx] += memory_per_layer_each_stage_no_recompute[0][stage_idx]
        return memory_cost_per_stage    
    
    def get_time_cost_for_specific_strategy(self, strategy:LayerWiseStrategy, global_batch_size:int, mixed_precision_type:str, accumulation_steps:int):
        args = self.args
        
        # get some basic information
        micro_batch_size = global_batch_size // accumulation_steps
            
        # calculate time cost for each type layer
        timecost_per_layer = [0 for _ in range(args.num_layertype)]
        timecost_per_layer_no_comm = [0 for _ in range(args.num_layertype)]
        sp_space = 'tp+sp'
        for i in range(args.num_layertype):
            # global_batch_size其实是除以了累积步数的
            time_cost_model_args = TimeCostModelArguments(strategy=strategy, global_batch_size=micro_batch_size, mixed_precision_type=mixed_precision_type, seq_length=self.seqlen_list[i], hidden_size=self.hidden_size_list[i],
                                                          forward_computation_time=self.time_profiled_list[i], parameter_memory=self.param_sizes[i], dp_overlap_coe=self.overlap_coe, bct_overlap_coe=self.overlap_coe, allreduce_coe_dict=self.allreduce_coe, p2p_coe_dict=self.p2p_coe, bct_fct_coe=2,
                                                          all2all_dict=self.sp_all2all, allreduce_dict=self.sp_allreduce, sp_space=sp_space)
            timecost_per_layer[i] = TimeCostModel(time_cost_model_args).gen_result()
            time_cost_model_args_no_comm = TimeCostModelArguments(strategy=strategy, global_batch_size=micro_batch_size, mixed_precision_type=mixed_precision_type, seq_length=self.seqlen_list[i], hidden_size=self.hidden_size_list[i],
                                                                forward_computation_time=self.time_profiled_list[i], parameter_memory=self.param_sizes[i], dp_overlap_coe=self.overlap_coe, bct_overlap_coe=self.overlap_coe, allreduce_coe_dict=self.allreduce_coe, p2p_coe_dict=self.p2p_coe, bct_fct_coe=2,
                                                                all2all_dict=self.sp_all2all, allreduce_dict=self.sp_allreduce, sp_space=sp_space,
                                                                no_comm=True)
            timecost_per_layer_no_comm[i] = TimeCostModel(time_cost_model_args_no_comm).gen_result()
        
        # calculate time cost for other layers
        other_time_cost_model_args = OtherTimeCostModelArguments(pp_size=strategy.pp_size, min_tp_size=strategy.tp_size, max_tp_size=strategy.tp_size, world_size=args.profile_gpu_num, 
                                                                 embed_sdp=1 if strategy.sharding_stage==3 else 0,
                                                                 hidden_size=self.hidden_size_list[0], mixed_precision_type=mixed_precision_type, 
                                                                 micro_batch_size=micro_batch_size // strategy.dp_size, # NOTE 此处需要修改为再除以在min_tp下的dp
                                                                 sequence_length_list=self.seqlen_list,
                                                                 other_memory_pp_off=self.other_memory_pp_off, other_memory_pp_on=self.other_memory_pp_on, other_time_profiled=self.other_time_profiled_list[0],
                                                                 allreduce_coe_dict=self.allreduce_coe, bct_fct_coe=2, dp_overlap_coe=self.overlap_coe,
                                                                vocab_use_ulysees=strategy.use_ulysses, sp_space=sp_space, allreduce_dict=self.sp_allreduce)
        time_other, time_other_no_comm = OtherTimeCostModel(other_time_cost_model_args).gen_result()  # len(time_other) == strategy.pp_size
        
        print(f'\tTime cost for each layer type: {timecost_per_layer}')
        print(f'\tTime cost for each layer type without communication: {timecost_per_layer_no_comm}')
        print(f'\tTime cost for other layers: {time_other}')
        print(f'\tTime cost for other layers without communication: {time_other_no_comm}')
        
        if strategy.pp_size == 1:
            cost = (timecost_per_layer_no_comm[0] * self.layernum_list[0] + time_other_no_comm[strategy.tp_size][0]) * \
                     (accumulation_steps - 1) + \
                    timecost_per_layer[0] * self.layernum_list[0] + \
                    time_other[strategy.tp_size][0]
            return cost
        
        # calculate the time cost of each stage
        pp_size = strategy.pp_size
        total_layer_num = sum(self.layernum_list)
        avg_layer_num = total_layer_num // pp_size
        last_stage_layer_num = total_layer_num - avg_layer_num * (pp_size - 1)
        layers = []
        for i in range(args.num_layertype):
            layer_num = self.layernum_list[i]
            layers.extend([i] * layer_num)
        stage_layers = [layers[i * avg_layer_num: (i + 1) * avg_layer_num] for i in range(pp_size - 1)] + [layers[-last_stage_layer_num:]]

        stage_costs_bsz_chunked = [0 for _ in range(pp_size)]
        stage_costs_compute = [0 for _ in range(pp_size)]
        for stage_idx in range(pp_size):
            for layer_type in stage_layers[stage_idx]:
                stage_costs_bsz_chunked[stage_idx] += timecost_per_layer[layer_type]
                stage_costs_compute[stage_idx] += timecost_per_layer_no_comm[layer_type]
        for stage_idx in range(pp_size):
            stage_costs_compute[stage_idx] += time_other_no_comm[strategy.tp_size][stage_idx]
            stage_costs_bsz_chunked[stage_idx] += time_other[strategy.tp_size][stage_idx]

        # compose all stages' time cost
        stage_costs_reduce = [total for total in stage_costs_bsz_chunked]
        result = np.sum(stage_costs_compute) + stage_costs_compute[-1] * (accumulation_steps - 1)
        print(f'pp_size: {pp_size}, tp_size: {strategy.tp_size}, time cost: {result}')
        result = max(result,
                     max(min(pp_size - 1, accumulation_steps - 1) * stage_costs_compute[0] * 1/3, np.sum(stage_costs_compute[1:]) * 1/3) + 
                     max(min(pp_size -1 , accumulation_steps - 1) * stage_costs_compute[0] * 2/3, np.sum(stage_costs_compute[1:]) * 2/3) +
                     stage_costs_compute[0] * max(0, accumulation_steps + 1 - pp_size))
        print(f'[linguangming] stage_costs_compute is {stage_costs_compute}')
        print(f'[linguangming] origin stage_costs_reduce is {stage_costs_reduce}')
        # for stage_idx in range(pp_size):
        #     stage_costs_reduce[stage_idx] -= np.sum(stage_costs_compute[:stage_idx + 1]) # sum不是很对把
        for stage_idx in range(pp_size):
            stage_costs_reduce[stage_idx] -= stage_costs_compute[stage_idx]
        print(f'[linguangming] after reduce stage_costs_reduce is {stage_costs_reduce}')
        reduce_time = np.sum(stage_costs_reduce)
        reduce_time = reduce_time if reduce_time > 0 else 0.0
        result += reduce_time
        
        return result
        
    # =================Utils Functions=================
    def strkeys2intkeys(self, config):
        # recursively convert
        if isinstance(config, dict):
            new_dict = {}
            for key, value in config.items():
                if isinstance(key, str) and key.isdigit():
                    new_dict[int(key)] = self.strkeys2intkeys(value)
                else:
                    new_dict[key] = self.strkeys2intkeys(value)
            return new_dict
        return config