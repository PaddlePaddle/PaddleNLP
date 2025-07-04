from ..utils import Strategy
from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Union, List

@dataclass
class TimeCostModelArguments:
    # initialize from scripts, config files, command line args or manual code.
    strategy: Strategy = field(default=None, metadata={"help": "The strategy of the model."})
    global_batch_size: int = field(default=8, metadata={"help": "The global batch size of the model."})  # 这个地方应该是micro_batch_size
    mixed_precision_type: str = field(default='fp16', metadata={"help": "The mixed precision type of the model."})
    seq_length: int = field(default=1024, metadata={"help": "The sequence length of the model."})
    hidden_size: int = field(default=4096, metadata={"help": "The hidden size of the model."})
    
    # initialize from profile data
    forward_computation_time: Optional[Union[float, np.ndarray]] = field(default=0.0, metadata={"help": "The forward computation time of one layer."})
    parameter_memory: Optional[Union[float, np.ndarray]] = field(default=0.0, metadata={"help": "The parameter memory of the one layer."})
    dp_overlap_coe: float = field(default=1.0, metadata={"help": "The dp overlap coefficient."})
    bct_overlap_coe: float = field(default=1.0, metadata={"help": "The bct overlap coefficient"})
    allreduce_coe_dict: dict = field(default=None, metadata={"help": "The allreduce coefficient."})
    p2p_coe_dict: dict = field(default=None, metadata={"help": "The p2p coefficient dictionary."})
    bct_fct_coe: float = field(default=2.0, metadata={"help": "The bct fct coefficient."})
    
    # some fine tune args
    extra_overhead: float = field(default=0.0, metadata={"help": "The extra overhead of the model."})
    cost_model_coe: float = field(default=1.0, metadata={"help": "The cost model coefficient of the model."})
    
    # some args for debug or adjust
    no_comm: bool = field(default=False, metadata={"help": "Whether to disable communication."})
    dummy_layernum: int = field(default=24, metadata={"help": "The number of layers of the model."})  # just to scale the time cost, not the real number of layers.

class TimeCostModel:
    """     
        A class to estimate the time cost of one transformer layers, given:
            specific parallelization strategy, global batch size, sequence length.
    """
    def __init__(self, args:TimeCostModelArguments):
        self.args = args
        self.initialize()
        self.estimate_computation_time()
        self.estimate_dp_communication_cost()
        self.estimate_tp_communication_cost()
        self.estimate_pp_communication_cost()
    
    def initialize(self):
        args = self.args

        strategy = args.strategy
        self.pp_size = strategy.pp_size
        self.tp_size = strategy.tp_size
        self.dp_size = strategy.dp_size
        self.sharding_stage = strategy.sharding_stage
        self.recompute = strategy.recompute
        
        self.local_batch_size = args.global_batch_size // self.dp_size # NOTE It should be interpreted as `micro_batch_size` (per-GPU batch size).

    def estimate_computation_time(self):
        args = self.args
        if isinstance(args.forward_computation_time, np.ndarray): # when time profile-mode is batch or sequence, forward_computation time is popt meaning the linear function fitted parameters.
            def linear_func(x, m, c):
                return m * x + c
            self.fct = linear_func(self.local_batch_size / self.tp_size , args.forward_computation_time[0], args.forward_computation_time[1]) * args.dummy_layernum # / self.tp_size  # divide by tp_size because the time is profiled on ddp.
            # self.fct = linear_func(self.local_batch_size, args.forward_computation_time[0], args.forward_computation_time[1]) * args.dummy_layernum / self.tp_size  # divide by tp_size because the time is profiled on ddp.

        elif isinstance(args.forward_computation_time, float): # when time profile-mode is static, forward_computation_time is a float.
            self.fct = args.forward_computation_time * self.local_batch_size / self.tp_size * args.dummy_layernum

        self.bct = self.fct * args.bct_fct_coe
        if self.recompute:
            self.bct += self.fct
        
        # print(f'time cost model, fct:{self.fct}, bct:{self.bct}')
            
    def estimate_dp_communication_cost(self):
        args = self.args
        self.dp_message_size = 2 * (self.dp_size - 1) / self.dp_size * args.parameter_memory * args.dummy_layernum
        if args.mixed_precision_type == 'fp16' or args.mixed_precision_type == 'bf16':
            self.dp_message_size /= 2  # [NOTE] gradient is in fp16 or bf16, so the message size is halved.
            
        if args.no_comm:
            self.dp_message_size = 0.0
        
        self.fsdp_allgather_message_size = self.dp_message_size * 0.5
        
        self.dc = args.allreduce_coe_dict[self.dp_size]
    
    def estimate_tp_communication_cost(self):
        args = self.args
        tp_comm_times, fp32_bytes = 4, 4 # In tensor parallel: 2 comm steps for fwd, bwd. fp32 -> 4 bytes
        self.tp_message_size = 2 * (self.tp_size - 1) / self.tp_size * (self.local_batch_size * args.hidden_size * args.seq_length) * tp_comm_times * fp32_bytes * args.dummy_layernum / 1024 / 1024
        if args.mixed_precision_type == 'fp16' or args.mixed_precision_type == 'bf16':  # [NOTE] when paddle use mix_precision, the activation is still in fp32 此处应该按照level来进行判断
            self.tp_message_size /= 2
        if self.recompute:
            self.tp_message_size * 1.5
            
        self.tc = args.allreduce_coe_dict[self.tp_size]
        self.tp_communication_time = self.tp_message_size * self.tc
        # print(f'time cost model tp_message_size: {self.tp_message_size}, tp_communication_time: {self.tp_communication_time}')
    
    def estimate_pp_communication_cost(self):
        args = self.args
       
        if self.pp_size > 1:  
            fp32_bytes = 4
            self.p2p_message_size = self.pp_size * 2 * self.local_batch_size * args.hidden_size * args.seq_length * fp32_bytes / 1024 / 1024
            if args.mixed_precision_type == 'fp16' or args.mixed_precision_type == 'bf16':
                self.p2p_message_size /= 2
            self.pc = args.p2p_coe_dict[self.pp_size]
            self.p2p_communication_time = self.p2p_message_size * self.pc
        else:
            self.p2p_message_size = 0.0
            self.p2p_communication_time = 0.0
        # print(f'time cost model p2p_message_size: {self.p2p_message_size}, p2p_communication_time: {self.p2p_communication_time}')
    
    def bct_dp_overlap(self, dp_message_size, bct):
        args = self.args
        dp_overlap_time = dp_message_size * self.dc * args.dp_overlap_coe
        bct_overlap_time = bct * args.bct_overlap_coe
        
        if dp_overlap_time > bct_overlap_time:
            overlap_part = bct_overlap_time
            rest_part = (dp_message_size - bct_overlap_time / (self.dc * args.dp_overlap_coe)) * self.dc
            rest_dp_flag = True
        elif dp_overlap_time < bct_overlap_time:
            overlap_part = dp_overlap_time
            rest_part = bct - dp_overlap_time / args.bct_overlap_coe
            rest_dp_flag = False
        else:
            overlap_part = dp_overlap_time
            rest_part = 0.0
            rest_dp_flag = False
        # print(f'time cost model bct_dp_overlap: overlap_part: {overlap_part}, rest_part: {rest_part}, rest_dp_flag: {rest_dp_flag}')
        return overlap_part, rest_part, rest_dp_flag
    
    def gen_result(self):
        args = self.args
        
        if self.tp_size == 1 and self.dp_size > 1: # pp + dp
            overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct)
            result = self.fct + overlap_part + rest_part + args.extra_overhead
        elif self.dp_size == 1 and self.tp_size > 1: # pp + tp
            result = self.fct + self.bct + self.tp_communication_time + args.extra_overhead
        elif self.dp_size == 1 and self.tp_size == 1: # pure pp
            result = self.fct + self.bct
        else: # pp + tp + dp
            overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct)
            overall_overhead = self.fct + overlap_part + rest_part + self.tp_communication_time + args.extra_overhead
            result = overall_overhead
            # if self.tp_size < self.tp_size * self.dp_size // 2:
            #     overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct)
            #     overall_overhead = self.fct + overlap_part + rest_part + self.tp_communication_time + args.extra_overhead
            #     result = overall_overhead
            # else:
            #     overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct * 1 / 2)
            #     overall_overhead = self.fct + 1 / 2 * self.bct + overlap_part + rest_part + self.tp_communication_time + args.extra_overhead
            #     result = overall_overhead
                
        if self.sharding_stage == 3:
            result += self.fsdp_allgather_message_size * self.dc
        
        if self.pp_size > 1:
            result += self.p2p_communication_time
            
        coe = 0.001 * args.cost_model_coe  # seems to convert to milliseconds
        result = result * coe
        result = result / args.dummy_layernum
        return result
    
@dataclass
class OtherTimeCostModelArguments:
    # initialize from scripts, config files, command line args 
    pp_size: int = field(default=1, metadata={"help": "The pp size of the model."})
    min_tp_size: int = field(default=1, metadata={"help": "The min tp size of the model."})
    max_tp_size: int = field(default=1, metadata={"help": "The max tp size of the model."})
    world_size: int = field(default=1, metadata={"help": "The world size of the model."})
    sharding_stage: int = field(default=1, metadata={"help": "The sharding stage of the model."})
    
    hidden_size: int = field(default=4096, metadata={"help": "The hidden size of the model."})
    mixed_precision_type: str = field(default='fp16', metadata={"help": "The mixed precision type of the model."})
    micro_batch_size: int = field(default=8, metadata={"help": "The micro batch size of the model."})

    sequence_length_list: List[int] = field(default_factory=lambda: [1024], metadata={"help": "The sequence length list of the model."})

    other_memory_pp_off:dict = field(default_factory=lambda: {'model_states': 640, 'activation': 320})
    other_memory_pp_on:dict = field(default_factory=lambda: {'first_stage':{'model_states': 640, 'activation': 320}, 'last_stage':{'model_states': 640, 'activation': 320}})
    other_time_profiled: Optional[Union[float, np.ndarray]] = field(default=0.0, metadata={"help": "The other time profile of the model."})

    allreduce_coe_dict: dict = field(default=None, metadata={"help": "The allreduce coefficient."})
    bct_fct_coe: float = field(default=2, metadata={"help": "The bct fct coefficient."})
    dp_overlap_coe: float = field(default=1.0, metadata={"help": "The dp overlap coefficient."})
    
class OtherTimeCostModel:
    def __init__(self, args:OtherTimeCostModelArguments):
        self.args = args
        self.estimate_fct_time()
        self.estimate_dp_time()
        self.estimate_tp_time()
    
    def estimate_fct_time(self):
        args = self.args
        self.fct = {}
        
        tp_size = args.min_tp_size
        while tp_size <= args.max_tp_size and tp_size * args.pp_size <= args.world_size:
            dp_size = args.world_size // tp_size // args.pp_size
            if isinstance(args.other_time_profiled, np.ndarray): # when time profile-mode is batch or sequence, forward_computation time is popt meaning the linear function fitted parameters.
                def linear_func(x, m, c):
                    return m * x + c
                # fct_time = linear_func(args.micro_batch_size / dp_size / tp_size, args.other_time_profiled[0], args.other_time_profiled[1]) # / tp_size
                fct_time = linear_func(args.micro_batch_size / dp_size, args.other_time_profiled[0], args.other_time_profiled[1]) / tp_size

            else:
                fct_time = args.other_time_profiled * args.micro_batch_size // dp_size / tp_size
            
            if args.pp_size == 1: # no pp, just one stage
                self.fct[tp_size] = fct_time
            else: # pp, two stages, we assume the first stage and last stage have the same time cost.
                self.fct[tp_size] = (fct_time / 2, fct_time / 2)
            tp_size *= 2
        # print(f'othertime fct: {self.fct}')
    
    def estimate_dp_time(self):
        args = self.args
        self.dp_message_size = {}
        self.dp_coe = {}
        
        tp_size = args.min_tp_size
        while tp_size <= args.max_tp_size and tp_size * args.pp_size <= args.world_size:
            dp_size = args.world_size // tp_size // args.pp_size
            self.dp_coe[tp_size] = args.allreduce_coe_dict[dp_size] * (dp_size - 1) / dp_size
            if args.pp_size == 1:
                self.dp_message_size[tp_size] = args.other_memory_pp_off['model_states'][tp_size] / 4
            else:
                self.dp_message_size[tp_size] = (args.other_memory_pp_on['first_stage']['model_states'][tp_size] / 4, args.other_memory_pp_on['first_stage']['model_states'][tp_size] / 4)
            tp_size *= 2
            
        if args.sharding_stage == 3:
            self.fwd_factor = 0.5
            self.bwd_factor = 1.0
        else:
            self.fwd_factor = 0.0
            self.bwd_factor = 0.5
        
    def estimate_tp_time(self):
        args = self.args
        self.tp_time = {}
        
        tp_size = args.min_tp_size
        while tp_size <= args.max_tp_size and tp_size * args.pp_size <= args.world_size:
            dp_size = args.world_size // tp_size // args.pp_size
            tp_coe = args.allreduce_coe_dict[tp_size]
            mixed_precision_factor = 4 if args.mixed_precision_type == 'fp32' else 2
            # mixed_precision_factor = 4 # fixed to 4 when fp16_opt_level = 'O1'
            
            tp_message_size = []
            per_tp_message_time = []
            
            for seq_len in args.sequence_length_list:                
                tp_message_size.append((tp_size - 1) / tp_size * args.micro_batch_size / dp_size * seq_len * args.hidden_size / 1024 / 1024 * mixed_precision_factor)
                per_tp_message_time.append(tp_message_size[-1] * tp_coe)
            
            if args.pp_size == 1:
                self.tp_time[tp_size] = sum(per_tp_message_time) + per_tp_message_time[-1] # For T5 model (Actually, this code is not for T5 model, but for embedding + lmhead)
            else:
                self.tp_time[tp_size] = (per_tp_message_time[0], per_tp_message_time[-1])  # For T5 model, first stage and last stage have the same time cost. (Actually, this code is divide embedding and lmhead)
            
            tp_size *= 2
            
    def get_overlap_time(self, forward_comm_time, forward_comp_time, backward_comm_time, backward_comp_time, tp_time):
        forward_comp_time = forward_comp_time * self.args.dp_overlap_coe
        backward_comp_time = backward_comp_time * self.args.dp_overlap_coe
        if forward_comp_time > forward_comm_time:
            forward_time = forward_comm_time + (forward_comp_time - forward_comm_time) / self.args.dp_overlap_coe
        else:
            forward_time = forward_comm_time
        if backward_comp_time > backward_comm_time:
            backward_time = backward_comm_time + (backward_comp_time - backward_comm_time) / self.args.dp_overlap_coe
        else:
            backward_time = backward_comm_time
        return forward_time + backward_time + tp_time
            
    def gen_result(self):
        args = self.args
        other_time_cost = {}
        other_time_cost_no_comm = {}
        tp_size = args.min_tp_size
        
        for tp_size in self.dp_message_size.keys():
            other_time_cost[tp_size] = [0 for _ in range(args.pp_size)]
            other_time_cost_no_comm[tp_size] = [0 for _ in range(args.pp_size)]
            if args.pp_size == 1:
                other_time_cost[tp_size][0] = 0.001 * self.get_overlap_time(self.dp_message_size[tp_size] * self.dp_coe[tp_size] * self.fwd_factor, self.fct[tp_size], self.dp_message_size[tp_size] * self.dp_coe[tp_size] * self.bwd_factor, self.fct[tp_size] * args.bct_fct_coe, self.tp_time[tp_size])
                other_time_cost_no_comm[tp_size][0] = 0.001 * self.get_overlap_time(self.dp_message_size[tp_size] * self.dp_coe[tp_size] * self.fwd_factor, self.fct[tp_size], self.dp_message_size[tp_size] * self.dp_coe[tp_size] * (self.bwd_factor - 0.5), self.fct[tp_size] * args.bct_fct_coe, self.tp_time[tp_size])
            else:
                other_time_cost[tp_size][0] = 0.001 * self.get_overlap_time(self.dp_message_size[tp_size][0] * self.dp_coe[tp_size] * self.fwd_factor, self.fct[tp_size][0], self.dp_message_size[tp_size][0] * self.dp_coe[tp_size] * self.bwd_factor, self.fct[tp_size][0] * args.bct_fct_coe, self.tp_time[tp_size][0])
                other_time_cost[tp_size][-1] = 0.001 * self.get_overlap_time(self.dp_message_size[tp_size][-1] * self.dp_coe[tp_size] * self.fwd_factor, self.fct[tp_size][-1], self.dp_message_size[tp_size][-1] * self.dp_coe[tp_size] * self.bwd_factor, self.fct[tp_size][-1] * args.bct_fct_coe, self.tp_time[tp_size][-1])
                other_time_cost_no_comm[tp_size][0] = 0.001 * self.get_overlap_time(self.dp_message_size[tp_size][0] * self.dp_coe[tp_size] * self.fwd_factor, self.fct[tp_size][0], self.dp_message_size[tp_size][0] * self.dp_coe[tp_size] * (self.bwd_factor - 0.5), self.fct[tp_size][0] * args.bct_fct_coe, self.tp_time[tp_size][0])
                other_time_cost_no_comm[tp_size][-1] = 0.001 * self.get_overlap_time(self.dp_message_size[tp_size][-1] * self.dp_coe[tp_size] * self.fwd_factor, self.fct[tp_size][-1], self.dp_message_size[tp_size][-1] * self.dp_coe[tp_size] * (self.bwd_factor - 0.5), self.fct[tp_size][-1] * args.bct_fct_coe, self.tp_time[tp_size][-1])
                
        return other_time_cost, other_time_cost_no_comm