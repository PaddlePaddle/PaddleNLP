from dataclasses import dataclass, field
from ..utils import Strategy,  LayerWiseStrategy
import numpy as np

@dataclass
class MemoryCostModelArguments:
    strategy: LayerWiseStrategy = field(default=None, metadata={"help": "The strategy of the model."})
    global_batch_size: int = field(default=8, metadata={"help": "The global batch size of the model."})
    mixed_precision_type: str = field(default='fp16', metadata={"help": "The mixed precision type of the model."})
    stage_idx: int = field(default=0, metadata={"help": "The stage index of the model."})
    accumulation_steps: int = field(default=-1, metadata={"help": "The number of accumulation steps."})

    parameter_memory: float = field(default=0.0, metadata={"help": "The parameter memory of the model."})  # parameter memory of per layer(in MB) 
    tp_activation_per_bsz_dict:dict = field(default_factory=lambda: {1:85, 2:47, 4:28, 8:18.5})
 
class MemoryCostModel:
    def __init__(self, args:MemoryCostModelArguments):
        self.args = args
        self.validate()
        self.initialize()
        self.estimate_parameter_size()
        self.estimate_model_states_size()
        self.estimate_activation_size()
        
    def validate(self):
        args = self.args
        assert args.accumulation_steps > 0, f'Accumulation steps should be greater than 0, but got {args.accumulation_steps}'
    
    def initialize(self):
        args = self.args
        
        strategy = args.strategy
        self.pp_size = strategy.pp_size
        self.tp_size = strategy.tp_size
        self.dp_size = strategy.dp_size
        self.sharding_stage = strategy.sharding_stage # 此处的sharding_stage也是需要进行修改的
        self.recompute = strategy.recompute
        self.use_ulysses = strategy.use_ulysses
        
        self.local_batch_size = args.global_batch_size // self.dp_size
        microbatches = [t.shape[0] for t in chunk_like_torch(self.local_batch_size, args.accumulation_steps)]
        assert args.accumulation_steps == len(microbatches), f'Accumulation steps should be equal to the length of microbatches, but got {args.accumulation_steps} and {len(microbatches)}'
        end = self.pp_size - args.stage_idx if self.pp_size - args.stage_idx <= args.accumulation_steps else args.accumulation_steps
        self.act_1f1b_ratio = np.sum(microbatches[:end]) / np.sum(microbatches) if end > 0 else 0.0
        self.local_batch_size *= self.act_1f1b_ratio
        # print(f'local batch size: {self.local_batch_size}, act_1f1b_ratio: {self.act_1f1b_ratio}')

        # In PaddlePaddle, parameter gradients are stored in FP32 precision
        if args.accumulation_steps == 1:
            self.zero2_ratio = (lambda d: (1/7 + 6/7 * (1/d)))
            self.zero3_ratio = (lambda d: (1/d))
            
        else:
            self.zero2_ratio = (lambda d: (1/3 + 2/3 * (1/d)))
            self.zero3_ratio = (lambda d: (2/9 + 7/9 * (1/d)))
            
        if self.use_ulysses:
            self.sdp_size = self.dp_size * self.tp_size
        else:
            self.sdp_size = self.dp_size
            
    def estimate_parameter_size(self):
        args = self.args
        if self.use_ulysses:
            # when using Ulysees, the parameter size is not divided
            self.parameter_size = args.parameter_memory
        else:
            # when using tensor-parallelism, the parameter size is divided by the tensor parallel size (args.parameter_memory means the total parameter memory of one layer)
            self.parameter_size = args.parameter_memory / self.tp_size
        
    def estimate_model_states_size(self):
        if self.args.accumulation_steps == 1:
            self.model_states_size = self.parameter_size * 7
        else:
            self.model_states_size = self.parameter_size * 9
        if self.sharding_stage == 3:
            self.model_states_size *= self.zero3_ratio(self.sdp_size)
        elif self.sharding_stage == 2:
            self.model_states_size *= self.zero2_ratio(self.sdp_size)
    
    def estimate_activation_size(self):
        args = self.args
        if self.recompute:
            self.activation_size = args.tp_activation_per_bsz_dict['checkpoint'] * self.local_batch_size
            # NOTE adjust for sequence parallelism(Megatron-LM SP or Ulysees SP)
            self.activation_size /= self.tp_size 
        else:
            self.activation_size = args.tp_activation_per_bsz_dict[self.tp_size] * self.local_batch_size
        
    def get_memory_cost(self):
        result = {}
        result['parameter_size'] = self.parameter_size
        result['model_states'] = self.model_states_size
        result['activation'] = self.activation_size
        result['enc_total'] = self.model_states_size + self.activation_size
        # print(f'result: {result}')
        return result
    
@dataclass
class OtherMemoryCostModelArguments:
    min_tp_size: int = field(default=1, metadata={"help": "The minimum tp size of the model."})
    max_tp_size: int = field(default=8, metadata={"help": "The maximum tp size of the model."})
    world_size: int = field(default=8, metadata={"help": "The world size of the model."})
    pp_size: int = field(default=1, metadata={"help": "The pp size of the model."})
    sharding_stage: int = field(default=2, metadata={"help": "The sharding stage of the model."})
    global_batch_size: int = field(default=8, metadata={"help": "The global batch size of the model."})
    accumulation_steps: int = field(default=1, metadata={"help": "The number of accumulation steps."})
    mixed_precision_type: str = field(default='fp16', metadata={"help": "The mixed precision type of the model."})
    paddle_context_memory: float = field(default=0, metadata={"help": "The paddle context memory of the model."})
    other_memory_pp_off:dict = field(default_factory=lambda: {'model_states': 640, 'activation': 320})
    other_memory_pp_on:dict = field(default_factory=lambda: {'first_stage':{'model_states': 640, 'activation': 320}, 'last_stage':{'model_states': 640, 'activation': 320}})
    use_ulysses: bool = field(default=False, metadata={"help": "Whether to use Ulysees for memory cost estimation."})
    
class OtherMemoryCostModel:
    def __init__(self, args:OtherMemoryCostModelArguments):
        self.args = args
        self.initialize()
        self.estimate_memory_cost()
        
    def initialize(self):
        args = self.args

        if args.accumulation_steps == 1:
            self.zero2_ratio = (lambda d: (1/7 + 6/7 * (1/d)))
            self.zero3_ratio = (lambda d: (1/d))
        else:
            self.zero2_ratio = (lambda d: (1/3 + 2/3 * (1/d)))
            self.zero3_ratio = (lambda d: (2/9 + 7/9 * (1/d)))
            
        self.zero_ratio = self.zero2_ratio if args.sharding_stage == 2 else (self.zero3_ratio if args.sharding_stage == 3 else (lambda d: 1.0))
    
    def estimate_memory_cost(self):
        args = self.args
        
        tp_size_list, tp_size = [], args.min_tp_size
        while tp_size <= args.max_tp_size and tp_size * args.pp_size <= args.world_size:
            tp_size_list.append(tp_size)
            tp_size *= 2
        
        self.other_memory_cost = {}
        for tp_size in tp_size_list:
            dp_size = args.world_size // args.pp_size // tp_size
            tp_other_memory_cost = [0 for _ in range(args.pp_size)]
            other_layers_bsz = args.global_batch_size // dp_size // args.accumulation_steps  # already divided by accumulation steps

            if args.use_ulysses:
                model_tp = 1
                zero_ratio_value = self.zero_ratio(dp_size * tp_size)
            else:
                zero_ratio_value = self.zero_ratio(dp_size)
                model_tp = tp_size
            
            model_states_adjust = 1
            if args.accumulation_steps == 1:
                model_states_adjust = 7/9
            
            if args.pp_size == 1: # no pp -> only one stage
                # print("other cost model states", args.other_memory_pp_off['model_states'][tp_size] * self.zero_ratio(dp_size))
                # print("other cost model activation", args.other_memory_pp_off['activation'][tp_size] * other_layers_bsz)
                tp_other_memory_cost[0] = model_states_adjust * args.other_memory_pp_off['model_states'][model_tp] * zero_ratio_value + args.other_memory_pp_off['activation'][tp_size] * other_layers_bsz
                # print(f'other_model_states {model_states_adjust * args.other_memory_pp_off["model_states"][model_tp] * zero_ratio_value}')
                # print(f'other_activation {args.other_memory_pp_off["activation"][tp_size] * other_layers_bsz}')
            else: # pp -> 0:first stage, -1:last stage (here we assume accumulation_steps is greater than pp_size, which holds true in industrial practice. )
                other_layers_bsz_first = other_layers_bsz * args.pp_size
                other_layers_bsz_last = other_layers_bsz * 1
                # print(f'other_layers_bsz_first: {other_layers_bsz_first}, other_layers_bsz_last: {other_layers_bsz_last}')
                # print("other cost model states first stage", args.other_memory_pp_on['first_stage']['model_states'][tp_size] * self.zero_ratio(dp_size))
                # print("other cost model activation first stage", args.other_memory_pp_on['first_stage']['activation'][tp_size] * other_layers_bsz_first)
                # print("other cost model states last stage", args.other_memory_pp_on['last_stage']['model_states'][tp_size] * self.zero_ratio(dp_size))
                # print("other cost model activation last stage", args.other_memory_pp_on['last_stage']['activation'][tp_size] * other_layers_bsz_last)
                tp_other_memory_cost[0] = model_states_adjust* args.other_memory_pp_on['first_stage']['model_states'][model_tp] * zero_ratio_value + args.other_memory_pp_on['first_stage']['activation'][tp_size] * other_layers_bsz_first
                tp_other_memory_cost[-1] = model_states_adjust* args.other_memory_pp_on['last_stage']['model_states'][model_tp] * zero_ratio_value + args.other_memory_pp_on['last_stage']['activation'][tp_size] * other_layers_bsz_last
            
            for i in range(len(tp_other_memory_cost)):
                tp_other_memory_cost[i] += args.paddle_context_memory
            
            self.other_memory_cost[tp_size] = tp_other_memory_cost
            
    def get_other_memory_cost(self):
        return self.other_memory_cost
    
# some util functions
def chunk_like_torch(size, chunks):
    """Implement torch.arange(size).chunk(chunks) behavior using numpy"""
    if chunks <= 0:
        raise ValueError("chunks must be positive")
    
    chunk_size = (size + chunks - 1) // chunks  # ceiling division
    
    # Create splits
    splits = []
    for i in range(chunks):
        start = i * chunk_size
        if start >= size:
            break
        end = min(start + chunk_size, size)
        splits.append(np.arange(start, end))
    
    return splits