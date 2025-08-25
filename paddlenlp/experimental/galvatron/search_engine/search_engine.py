from ..utils import Strategy, LayerWiseStrategy
from dataclasses import dataclass, field
from ..cost_model.profile_data_parser import ProfileDataParser, ProfileDataParserArguments
import math
from typing import List
import copy
from .utils import ensure_log_dir, get_thread_logger
from .dynamic_programming import DpOnModel

@dataclass
class SearchEngineArguments:
    search_granularity: str = field(default="coarse-grained", metadata={"help": "The granularity of the search space."})
    world_size: int = field(default=8, metadata={"help": "The number of processes to use for distributed training."})

    min_bsz: int = field(default=64, metadata={"help": "The minimum batch size."})
    max_bsz: int = field(default=64, metadata={"help": "The maximum batch size."})
    bsz_step: int = field(default=1, metadata={"help": "The step size for batch size."})
    
    max_tp_size: int = field(default=8, metadata={"help": "The maximum tensor parallel size."})
    max_pp_size: int = field(default=8, metadata={"help": "The maximum pipeline parallel size."})
    
    mixed_precision_type: str = field(default="bf16", metadata={"help": "The mixed precision type to use."})
    memory_upper_limit: int = field(default=24, metadata={"help": "The upper limit of memory usage in GB"})
    
    sp_space: str = field(default="tp", metadata={"help": "The space for sharded parallelism, can be 'tp' or 'tp+sp'."})
    layernum: int = field(default=16, metadata={"help":"The layer num of model"})
    disable_sdp: int = field(default=0, metadata={"help": "Whether to disable sharded data parallelism."})
    disable_vtp: int = field(default=0, metadata={"help": "Whether to disable vocab tensor parallelism."})
    parallel_search: int = field(default=0, metadata={"help": "Whether to enable parallel search."})
    log_dir: str = field(default="./search-engine-logs", metadata={"help": "The directory to save logs."})
    
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
        self.sp_space = args_dict.get("--sp_space", self.sp_space)
        self.layernum = int(args_dict.get("--layernum", self.layernum))
        self.disable_sdp = int(args_dict.get("--disable_sdp", self.disable_sdp))
        self.disable_vtp = int(args_dict.get("--disable_vtp", self.disable_vtp))
        self.parallel_search = int(args_dict.get("--parallel_search", self.parallel_search))
        self.log_dir = args_dict.get("--log_dir", self.log_dir)
        
class SearchEngine:
    def __init__(self, args_dict):
        self.args = SearchEngineArguments()
        self.args.initialize(args_dict)        
        # 先注释掉
        parser_data_args = ProfileDataParserArguments()
        parser_data_args.initialize(args_dict)
        self.parser = ProfileDataParser(parser_data_args)

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
    
    def set_cost_model(self):
        self.memory_cost_model_args_dict = {
            'mixed_precision_type': self.args.mixed_precision_type,
            'parameter_memory': self.parser.param_sizes[0],
            'tp_activation_per_bsz_dict': self.parser.act_sizes[0]
        }
        self.other_memory_cost_model_args_dict = {
            'world_size': self.args.world_size,
            'mixed_precision_type': self.args.mixed_precision_type,
            'other_memory_pp_off': self.parser.other_memory_pp_off,
            'other_memory_pp_on': self.parser.other_memory_pp_on
        }
        self.time_cost_model_args_dict = {
            'mixed_precision_type' : self.args.mixed_precision_type,
            'seq_length': self.parser.seqlen_list[0],
            'hidden_size': self.parser.hidden_size_list[0],
            'forward_computation_time' : self.parser.time_profiled_list[0],
            'parameter_memory': self.parser.param_sizes[0],
            'dp_overlap_coe': self.parser.overlap_coe,
            'bct_overlap_coe': self.parser.overlap_coe,
            'allreduce_coe_dict': self.parser.allreduce_coe,
            'p2p_coe_dict': self.parser.p2p_coe,
            'bct_fct_coe': 2.0,
            'all2all_dict': self.parser.sp_all2all,
            'allreduce_dict': self.parser.sp_allreduce,
            'sp_space': self.args.sp_space,
        }
        self.other_time_cost_args_dict = {
            'world_size': self.args.world_size,
            'hidden_size': self.parser.hidden_size_list[0],
            'mixed_precision_type' : self.args.mixed_precision_type,
            'sequence_length_list': [self.parser.seqlen_list[0]], #  actually [self.parser.seqlen_list[0]] == self.parser.seqlen_list
            'other_memory_pp_off': self.parser.other_memory_pp_off,
            'other_memory_pp_on': self.parser.other_memory_pp_on,
            'other_time_profiled': self.parser.other_time_profiled_list[0],
            'allreduce_coe_dict':self.parser.allreduce_coe,
            'bct_fct_coe': 2.0,
            'dp_overlap_coe': self.parser.overlap_coe,
            'sp_space': self.args.sp_space,
            'allreduce_dict': self.parser.sp_allreduce,
        }
        print(f'memory_cost_model_args_dict: {self.memory_cost_model_args_dict}')
        print(f'other_memory_cost_model_args_dict: {self.other_memory_cost_model_args_dict}')
        print(f'time_cost_model_args_dict: {self.time_cost_model_args_dict}')
        print(f'other_time_cost_args_dict: {self.other_time_cost_args_dict}')

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
        
    def generate_layerwise_strategies(self):
        args = self.args
        layerwise_strategy_set: List[LayerWiseStrategy] = []
        
        strategy_template = LayerWiseStrategy()
        
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
                sharding_stage_set = [0, 2, 3] if dp_size > 1 else [0]
                for sharding_stage in sharding_stage_set:
                    for recompute in [0, 1]:
                        strategy_template.pp_size = pp_size
                        strategy_template.tp_size = tp_size
                        strategy_template.dp_size = dp_size
                        strategy_template.sharding_stage = sharding_stage
                        strategy_template.recompute = recompute
                        layerwise_strategy_set.append(copy.deepcopy(strategy_template))

        if args.sp_space == 'tp':
            for strategy in layerwise_strategy_set:
                strategy.use_ulysses = 0
        elif args.sp_space == 'tp+sp':
            for strategy in layerwise_strategy_set:
                strategy.use_ulysses = 0
            strategy_set_copy = copy.deepcopy(layerwise_strategy_set) 
            for strategy in strategy_set_copy:
                strategy.use_ulysses = 1
                layerwise_strategy_set.append(strategy)
        else:
            raise NotImplementedError(f"SP space '{args.sp_space}' is not implemented.")   

        self.layerwise_strategies = layerwise_strategy_set
        print(f'[galvatron/search_engine/search_engine.py] layerwise_strategies has initialized, with {len(layerwise_strategy_set)} strategies.')
    
    def layerwise_parallelism_optimization(self):
        print('='*25, 'Galvatron Search Engine Start Searching', '='*25)
        
        print('-----', '[Searching Memory Info]', 'Memory constraint:', self.args.memory_upper_limit * 1024, 'MB', '-----')

        args = self.args
        
        results = dict()
        temp_strategies = copy.deepcopy(self.layerwise_strategies)
        max_throughput, optimal_bsz = -1, -1
        
        # generate total_min_tp and total_max_tp
        total_min_tp, i = [], 1
        while i <= args.world_size and i <= args.max_tp_size:
            total_min_tp.append(i)
            i *= 2
            
        if args.disable_vtp:
            total_min_tp = [1]
        
        total_max_tp = total_min_tp.copy()
        
        # generate sp_search_space and total_vsp
        if args.sp_space == 'tp':
            total_vsp = [0]
            sp_search_space = ['tp-only']
        elif args.sp_space == 'tp+sp':
            total_vsp = [0, 1]
            sp_search_space = ['tp-only', 'sp-only', 'tp-and-sp']
        else:
            raise NotImplementedError(f"SP space '{args.sp_space}' is not implemented.")
        
        # generate total_embed_sdp
        if args.disable_sdp:
            total_embed_sdp = [0]
        else:
            total_embed_sdp = [0, 1]
            
        # define the search_for_chunk function # TODO 修改这个函数的命名
        def search_for_chunk(bsz, accumulation_steps, min_tp, max_tp, vsp, embed_sdp):
            # log_dir = self.args.log_dir + '/%s_%dnodes_%dgpus_%dGB'%(self.model_name, self.args.num_nodes, self.args.num_gpus_per_node, self.memory_constraint//1024)
            log_dir = self.args.log_dir + '/log'
            logger = get_thread_logger(bsz, accumulation_steps, min_tp, max_tp, vsp, embed_sdp, log_dir)
            log_dir = ensure_log_dir(log_dir)
            logger.info(f"Starting search for bsz={bsz}, accumulation_steps={accumulation_steps}, min_tp={min_tp}, max_tp={max_tp}, vsp={vsp}, embed_sdp={embed_sdp}")

            result = dict()
            for sp_search in sp_search_space:
                if sp_search == 'tp-only' and vsp == 1:
                    logger.info(f'Skipping tp-only search for vsp={vsp}')
                    continue
                if sp_search == 'sp-only' and vsp == 0:
                    logger.info(f'Skipping sp-only search for vsp={vsp}')
                    continue
                
                # filter 
                strategies = [s for s in temp_strategies if min_tp <= s.tp_size and s.tp_size <= max_tp] # tp_size belongs to [min_tp, max_tp]
                if len(strategies) == 0:
                    logger.info(f'no strategy satisfies the constraints [min_tp,  max_tp]')
                    continue

                strategies = [s for s in strategies if bsz // accumulation_steps >= args.world_size // s.pp_size // min_tp] # micro_batch_size (micro_batch_size = global_batch_size // accumulation_steps) should greater than max_dp_size(max_dp_size = world_size // pp_size // min_tp)
                if len(strategies) == 0:
                    logger.info(f'no strategy satisfies the constraints [bsz // accumulation_steps >= args.world_size // s.pp_size // min_tp]')
                    continue

                if sp_search == 'tp-only':
                    strategies = [s for s in strategies if s.use_ulysses == 0]
                elif sp_search == 'sp-only':
                    strategies = [s for s in strategies if s.use_ulysses == 1]
                if len(strategies) == 0:
                    logger.info(f'no strategy satisfies the constraints [{sp_search}]')
                    continue
                
                # get all possible pp_size and filter 
                pp_deg_list = sorted(list(set(s.pp_size for s in strategies)))
                pp_deg_list = [pp for pp in pp_deg_list if pp * min_tp <= args.world_size and bsz % (args.world_size // pp // min_tp) == 0]
                if len(pp_deg_list) == 0:
                    logger.info(f'no strategy satisfies the constraints [pp * min_tp <= args.world_size and bsz % (args.world_size // pp // min_tp) == 0]')
                    continue
                
                strategies = [s for s in strategies if s.pp_size in pp_deg_list]
                
                # Calculate the micro-batch size at min_tp under different pp_deg values
                mbsz_dict = dict()
                for pp in pp_deg_list:
                    mbsz_dict[pp] = (bsz // (args.world_size // pp // min_tp) + accumulation_steps - 1) // accumulation_steps
                # strict mode: search accumulation_steps must be equal to real accumulation_steps 
                strategies = [s for s in strategies if accumulation_steps == (bsz // (args.world_size // s.pp_size // min_tp) + mbsz_dict[s.pp_size] - 1) // mbsz_dict[s.pp_size]]
                if len(strategies) == 0:
                    logger.info(f'no strategy satisfies the constraints [accumulation_steps == (bsz // (args.world_size // s.pp_size // min_tp) + mbsz_dict[s.pp_size] - 1) // mbsz_dict[s.pp_size]]. Actually, this situation not happen')
                    continue
                
                # get pp_stage_dict
                pp_deg_list = sorted(list(set(s.pp_size for s in strategies)))
                pp_stage_dict = get_pp_stage(pp_deg_list, args.layernum)
                
                # dynamic programming solve
                result[sp_search] = self.dynamic_programming(strategies, bsz, accumulation_steps, mbsz_dict, pp_stage_dict, min_tp, max_tp, vsp, embed_sdp, sp_search, logger)
                result[sp_search]['pp_stage_dict'] =  copy.deepcopy(pp_stage_dict)
            return result
                
        # start searching
        if args.parallel_search: # parallel search to speed up the search # TODO 
            pass
        else:
            for bsz in self.BSZs:
                results[bsz] = dict()
                accumulation_steps_list = range(1, bsz + 1)
                for accumulation_steps in accumulation_steps_list:
                    results[bsz][accumulation_steps] = dict()
                    if bsz % accumulation_steps != 0: # skip the accumulation steps that cannot divide the batch size
                        continue
                    for min_tp in total_min_tp:
                        results[bsz][accumulation_steps][min_tp] = dict()
                        for max_tp in total_max_tp:
                            results[bsz][accumulation_steps][min_tp][max_tp] = dict()
                            if min_tp > max_tp:
                                continue
                            for vsp in total_vsp:
                                results[bsz][accumulation_steps][min_tp][max_tp][vsp] = dict()
                                for embed_sdp in total_embed_sdp:
                                    print(f'Start processing: Batch Size: {bsz}, Accumulation Steps: {accumulation_steps}, Min TP: {min_tp}, Max TP: {max_tp}, VSP: {vsp}, Embed SDP: {embed_sdp}', flush=True)
                                    results[bsz][accumulation_steps][min_tp][max_tp][vsp][embed_sdp] = search_for_chunk(bsz, accumulation_steps, min_tp, max_tp, vsp, embed_sdp)
        
        # store the results
        for bsz in results:
            for accumulation_steps in results[bsz]:
                for min_tp in results[bsz][accumulation_steps]:
                      for max_tp in results[bsz][accumulation_steps][min_tp]:
                          for vsp in results[bsz][accumulation_steps][min_tp][max_tp]:
                              for embed_sdp in results[bsz][accumulation_steps][min_tp][max_tp][vsp]:
                                  for sp_search in results[bsz][accumulation_steps][min_tp][max_tp][vsp][embed_sdp]:
                                        throughput = results[bsz][accumulation_steps][min_tp][max_tp][vsp][embed_sdp][sp_search]['throughput']
                                        pp_stage_dict = results[bsz][accumulation_steps][min_tp][max_tp][vsp][embed_sdp][sp_search]['pp_stage_dict']
                                        if throughput > max_throughput:
                                            max_throughput = throughput
                                            optimal_bsz = bsz
                                            optimal_chunk = accumulation_steps
                                            optimal_min_tp = min_tp
                                            optimal_max_tp = max_tp
                                            optimal_vsp = vsp
                                            optimal_embed_sdp = embed_sdp
                                            optimal_sp_search = sp_search
                                            optimal_pp_stage_dict = pp_stage_dict 
        if max_throughput > 0:
            print('\nFinal results of max memory %d GB:'%self.args.memory_upper_limit)
            re = results[optimal_bsz][optimal_chunk][optimal_min_tp][optimal_max_tp][optimal_vsp][optimal_embed_sdp][optimal_sp_search]
            re['vsp'] = optimal_vsp
            re['embed_sdp'] = optimal_embed_sdp
            print(f"Optimal bsz = {optimal_bsz} Optimal chunk = {optimal_chunk} Optimal vocab tp = {re['vtp']} Optimal vocab sp = {optimal_vsp} Optimal embed sdp = {optimal_embed_sdp} Max throughput={re['throughput']} samples/s")
            print(f"pp_deg={re['min_pp_deg']} Minimized timecost={re['min_cost']} Memory remaining={re['mem_remain']} Memory cost={re['mem_cost']}")
            print(f"Min_tp={optimal_min_tp} Max_tp={optimal_max_tp} ")
            
            print(f'[linguangming] re[min_res_list]')
            for item in re['min_res_list']:
                if isinstance(item, List):
                    for sub_item in item:
                        print(sub_item)
                # print(item)
            config = {}
            config['pp_size'] = re['min_pp_deg']
            config['vtp'] =  re['vtp']
            config['vsp_flag'] = re['vsp']
            config['embed_sdp'] = re['embed_sdp']
            config['global_batch_size'] = optimal_bsz
            config['gradient_accumulation_steps'] = optimal_chunk
            
            dp_size_list, sharding_stage_list, tp_size_list, usp_flag_list, recompute_list = [], [], [], [], []
            for item in re['min_res_list']:
                if isinstance(item, List):
                    for sub_item in item:
                        dp_size_list.append(sub_item.dp_size)
                        sharding_stage_list.append(sub_item.sharding_stage)
                        tp_size_list.append(sub_item.tp_size)
                        usp_flag_list.append(sub_item.use_ulysses)
                        recompute_list.append(sub_item.recompute)
            config['dp_size_list'] = ','.join(str(dp_size) for dp_size in dp_size_list)
            config['sharding_stage_list'] = ','.join(str(sharding_stage) for sharding_stage in sharding_stage_list)
            config['tp_size_list'] = ','.join(str(tp_size) for tp_size in tp_size_list)
            config['usp_flag_list'] = ','.join(str(usp_flag) for usp_flag in usp_flag_list)
            config['recompute_list'] = ','.join(str(recompute) for recompute in recompute_list)
            
            ave_num_layer = self.args.layernum // config['pp_size']
            last_num_layer = self.args.layernum - (config['pp_size'] - 1) * ave_num_layer
            pp_stage_idx_list = []
            for i in range(config['pp_size']):
                if i == config['pp_size'] - 1:
                    pp_stage_idx_list.extend([i] * last_num_layer)
                else:
                    pp_stage_idx_list.extend([i] * ave_num_layer)
            config['pp_stage_idx_list'] = ','.join(str(pp_stage_idx) for pp_stage_idx in pp_stage_idx_list)
            
            store_path = './configs/fine_grained_config.json'
            import os, json
            if os.path.exists(os.path.dirname(store_path)) == False:
                os.makedirs(os.path.dirname(store_path))
            with open(store_path, 'w') as fp:
                json.dump(config, fp, indent=4)
        else:
            print("No valid configuration found.")
        
        print("-----------------------------------------")
        print('='*25, 'Galvatron Search Engine End Searching','='*25)

        return max_throughput
                              
    def dynamic_programming(self, strategies:List[LayerWiseStrategy], bsz, accumulation_steps, mbsz_dict, pp_stage_dict, min_tp, max_tp, vsp, embed_sdp, sp_search, logger):
        logger.info('\n')
        args = self.args
        logger.info(f'bsz={bsz} pp_stage_dict{pp_stage_dict}')
        dp_on_model = DpOnModel(strategies_set=strategies, 
                                world_size=args.world_size, 
                                pp_stage_dict=pp_stage_dict, 
                                memory_cost_model_args_dict=self.memory_cost_model_args_dict,
                                other_memory_cost_model_args_dict=self.other_memory_cost_model_args_dict,
                                time_cost_model_args_dict=self.time_cost_model_args_dict,
                                other_time_cost_args_dict=self.other_time_cost_args_dict,
                                fine_grained=True if args.search_granularity == 'fine-grained' else False,
                                layer_num=args.layernum, 
                                max_mem=args.memory_upper_limit * 1024, 
                                mem_cache_flag=True, 
                                sequence_length=self.parser.seqlen_list[0],
                                hidden_size=self.parser.hidden_size_list[0],
                                mixed_precision=args.mixed_precision_type,
                                allreduce_coe_dict=self.parser.allreduce_coe,
                                logger=logger)
        logger.info(f"****Searching with bsz={bsz} accumulation_steps={accumulation_steps} min_tp={min_tp} max_tp={max_tp} vsp={vsp} embed_sdp={embed_sdp} sp_search={sp_search}****")
        logger.info(f'Mbsz_dict for bsz {bsz}: {mbsz_dict}')
        
        min_cost, min_res_list, min_pp_deg, mem_remain, mem_cost, min_vtp = dp_on_model.fit(bsz, min_tp, max_tp, vsp, embed_sdp, sp_search, mbsz_dict = mbsz_dict)
        throughput = bsz / min_cost
        logger.info(f"[Optimal pp_deg={min_pp_deg}] Minimized timecost={min_cost} Memory remaining={mem_remain} Memory cost={mem_cost} Vocab tp={min_vtp}")
        logger.info(f"Max throughput={throughput} samples/s")
        logger.info('\n')
        
        result = {'min_cost': min_cost, 'min_res_list': min_res_list, 'min_pp_deg': min_pp_deg, 
                        'mem_remain': mem_remain, 'mem_cost': mem_cost, 'throughput': throughput, "vtp": min_vtp}
        return result
        
                                    
def get_pp_stage(pp_deg_list, layernum):
    pp_stage_dict = dict()
    for pp_deg in pp_deg_list:
        pp_stage_dict[pp_deg] = pp_division_even(pp_deg, layernum)
    return pp_stage_dict

def pp_division_even(pp_deg, layernum):
    avg_layer_num = layernum // pp_deg
    last_layer_num = layernum - avg_layer_num * (pp_deg - 1)
    pp_division = [avg_layer_num] * (pp_deg - 1) + [last_layer_num]
    return pp_division
    