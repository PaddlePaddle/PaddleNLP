from ..utils import Strategy
from dataclasses import dataclass, field
from ..cost_model.profile_data_parser import ProfileDataParser, ProfileDataParserArguments
import math
from typing import List

@dataclass
class SearchEngineArguments:
    search_granularity: str = field(default="coarse-grained", metadata={"help": "The granularity of the search space."})
    world_size: int = field(default=8, metadata={"help": "The number of processes to use for distributed training."})

    min_bsz: int = field(default=64, metadata={"help": "The minimum batch size."})
    max_bsz: int = field(default=64, metadata={"help": "The maximum batch size."})
    bsz_step: int = field(default=1, metadata={"help": "The step size for batch size."})
    
    max_tp_size: int = field(default=8, metadata={"help": "The maximum tensor parallel size."})
    max_pp_size: int = field(default=8, metadata={"help": "The maximum pipeline parallel size."})
    
    mixed_precision_type: str = field(default="bp16", metadata={"help": "The mixed precision type to use."})
    memory_upper_limit: int = field(default=24, metadata={"help": "The upper limit of memory usage in GB"})
    
    def initialize(self, args_dict):
        self.search_granularity = args_dict.get("--search_granularity", self.search_granularity)
        self.world_size = int(args_dict.get("--world_size", self.world_size))
        self.min_bsz = int(args_dict.get("--min_bsz", self.min_bsz))
        self.max_bsz = int(args_dict.get("--max_bsz", self.max_bsz))
        self.bsz_step = int(args_dict.get("--bsz_step", self.bsz_step))
        self.max_tp_size = int(args_dict.get("--max_tp_size", self.max_tp_size))
        self.max_pp_size = int(args_dict.get("--max_pp_size", self.max_pp_size))
        self.mixed_precision_type = args_dict.get("--mixed_precision_type", self.mixed_precision_type)
        self.memory_upper_limit = int(args_dict.get("--memory_upper_limit", self.memory_upper_limit))
        
class SearchEngine:
    def __init__(self, args_dict):
        self.args = SearchEngineArguments()
        self.args.initialize(args_dict)
        
        parser_data_args = ProfileDataParserArguments()
        parser_data_args.initialize(args_dict)
        self.parser = ProfileDataParser(parser_data_args)
        
        self.generate_strategies()
        self.set_searching_bsz()

    def generate_strategies(self):
        args = self.args
        
        self.strategy_set:List[Strategy] = []
        
        i, degree_set = 1, []
        while i <= args.world_size:
            degree_set.append(i)
            i *= 2
        
        for pp_size in degree_set:
            if pp_size > args.max_pp_size:
                continue
            for tp_size in degree_set:
                if pp_size * tp_size > args.world_size:
                    continue
                if tp_size > args.max_tp_size:
                    continue
                dp_size = args.world_size // (pp_size * tp_size)
                # sharding_stage_set = [0, 2, 3] if dp_size > 1 else [0]
                sharding_stage_set = [0, 2] if dp_size > 1 else [0] # when in static mode, RuntimeError: Operation((%0) = "pd_op.embedding_grad" is not support sharded by shard_tensor op in pir mode happend.
                for recompute in [0, 1]:
                    for sharding_stage in sharding_stage_set:
                        strategy = Strategy(pp_size=pp_size, tp_size=tp_size, dp_size=dp_size, sharding_stage=sharding_stage, recompute=recompute)
                        self.strategy_set.append(strategy)
                            
        print(f'SearchEngine strategt_set: {self.strategy_set}')
    
    def set_searching_bsz(self):
        args = self.args
        min_bsz, max_bsz, bsz_step = args.min_bsz, args.max_bsz, args.bsz_step
        min_bsz = max(min_bsz, bsz_step)
        min_bsz = min_bsz // bsz_step * bsz_step
        max_bsz = int(math.ceil(max_bsz / bsz_step) * bsz_step) if max_bsz % bsz_step != 0 else max_bsz + bsz_step
        self.BSZs = list(range(min_bsz, max_bsz, bsz_step))
        
        # change the min_bsz and max_bsz
        args.min_bsz = min_bsz
        args.max_bsz = max_bsz
        
        print('-----', '[Searching Batch Sizes Info]', 'Min bsz:', args.min_bsz, 'Max bsz:', args.max_bsz, 'bsz_step:', args.bsz_step, '-----')
        print('Searching Batch Sizes:', self.BSZs)
      
    def parallelism_optimization(self):
        args = self.args
        
        if args.search_granularity == 'coarse-grained':
            optimal_solution, max_throughput, optimal_history = {}, -1, []
            results = dict()
            for bsz in self.BSZs:
                results[bsz] = dict()
                accumulation_steps_list = range(1, bsz + 1)
                for accumulation_steps in accumulation_steps_list:
                    results[bsz][accumulation_steps] = dict()
                    if bsz % accumulation_steps != 0:
                        continue
                    for strategy in self.strategy_set:
                        results[bsz][accumulation_steps][strategy.serialize()] = dict()
                        if bsz // accumulation_steps < strategy.dp_size:
                            continue
                        memory_cost = self.parser.get_memory_cost_for_specific_strategy(strategy, bsz, args.mixed_precision_type, accumulation_steps)
                        time_cost = self.parser.get_time_cost_for_specific_strategy(strategy, bsz, args.mixed_precision_type, accumulation_steps)
                        results[bsz][accumulation_steps][strategy.serialize()]['memory_cost'] = memory_cost
                        results[bsz][accumulation_steps][strategy.serialize()]['time_cost'] = time_cost
                        results[bsz][accumulation_steps][strategy.serialize()]['throughput'] = bsz / time_cost if time_cost > 0 else 0
                        results[bsz][accumulation_steps][strategy.serialize()]['OOM'] = memory_cost[0] > args.memory_upper_limit * 1024 # memory_cost[0] means the first stage memory cost
                        if results[bsz][accumulation_steps][strategy.serialize()]['throughput'] > max_throughput and not results[bsz][accumulation_steps][strategy.serialize()]['OOM']:
                            max_throughput = results[bsz][accumulation_steps][strategy.serialize()]['throughput']
                            optimal_solution = {
                                'bsz': bsz,
                                'accumulation_steps': accumulation_steps,
                                'strategy': strategy,
                                'memory_cost': memory_cost,
                                'time_cost': time_cost,
                                'throughput': max_throughput
                            }
                            optimal_history.append(optimal_solution)
                        print(f'Batch Size: {bsz}, Accumulation Steps: {accumulation_steps}, Strategy: {strategy.serialize()}, Memory Cost: {memory_cost} MB, Time Cost: {time_cost} s, Throughput: {results[bsz][accumulation_steps][strategy.serialize()]["throughput"]} Sample/s, OOM: {results[bsz][accumulation_steps][strategy.serialize()]["OOM"]}')
            print('-----', '[Optimal Solution History]', '-----')
            for history in optimal_history:
                print(f'Batch Size: {history["bsz"]}, Accumulation Steps: {history["accumulation_steps"]}, Strategy: {history["strategy"].serialize()}, Memory Cost: {history["memory_cost"]} MB, Time Cost: {history["time_cost"]} s, Throughput: {history["throughput"]} Sample/s')
            print('-----', '[Optimal Solution]', '-----')
            print('Optimal Solution:', optimal_solution)
            
            import os 
            current_dir = os.getcwd()
            optimal_solution_path = os.path.join(current_dir, './configs/optimal_solution.json')
            with open(optimal_solution_path, 'w') as f:
                import json
                info = {
                    'bsz': optimal_solution['bsz'],
                    'accumulation_steps': optimal_solution['accumulation_steps'],
                    'strategy': optimal_solution['strategy'].serialize(),
                    'memory_cost': optimal_solution['memory_cost'],
                    'time_cost': optimal_solution['time_cost'],
                    'throughput': optimal_solution['throughput']
                }
                json.dump(info, f, indent=4)
            return results, optimal_solution
        else:
            raise NotImplementedError(f"Search granularity '{args.search_granularity}' is not implemented.")