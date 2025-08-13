import sys, os, copy
import numpy as np
from typing import List
from ..utils import LayerWiseStrategy
from ..cost_model.time_cost_model import TimeCostModelArguments, OtherTimeCostModelArguments, TimeCostModel, OtherTimeCostModel, pipeline_costmodel
from ..cost_model.memory_cost_model import MemoryCostModel, OtherMemoryCostModel, MemoryCostModelArguments, OtherMemoryCostModelArguments

class DPAlg:
    def __init__(self, 
                 max_mem=8200, 
                 other_mem_cost=None, 
                 other_time_cost = None, 
                 layer_num=24, 
                 strategy_num=4, 
                 strategy_set=None, 
                 fine_grained_mode=True, 
                 use_cpp_core=True) -> None:
        assert(other_mem_cost != None)
        self.max_mem = max_mem + 1
        self.layer_num = layer_num
        self.strategy_num = strategy_num
        self.other_mem_cost = other_mem_cost
        self.other_time_cost = other_time_cost

        self._f = np.full((self.max_mem, strategy_num), 0, dtype=np.float64)
        
        self.v_data = None
        self.inter_cost = None
        self.intra_cost = None

        self._mark = np.full((layer_num, self.max_mem, strategy_num), -1, dtype=np.int32)
        self.use_cpp_core = use_cpp_core
        self.strategy_set = strategy_set
        self.fine_grained_mode = fine_grained_mode
    
    def set_v_and_cost(self, v: np.ndarray, intra_layer_cost: np.ndarray, inter_layer_cost: np.ndarray):
        assert v.ndim == 2
        assert inter_layer_cost.ndim == 3
        assert intra_layer_cost.ndim == 2

        assert v.shape[0] == self.layer_num
        assert v.shape[1] == self.strategy_num

        assert inter_layer_cost.shape[0] == self.layer_num
        assert inter_layer_cost.shape[1] == self.strategy_num and inter_layer_cost.shape[2] == self.strategy_num

        assert intra_layer_cost.shape[0] == self.layer_num
        assert intra_layer_cost.shape[1] == self.strategy_num

        self.v_data = v.astype(np.int32)
        self.inter_cost = inter_layer_cost
        self.intra_cost = intra_layer_cost

    def fit(self):
        if not self.fine_grained_mode:
            res_list = {k:np.full((self.layer_num), -1, dtype=np.int32) for k,v in self.other_mem_cost.items()}
            total_cost = {k:np.inf for k,v in self.other_mem_cost.items()}
            remaining_mem = {k:-1 for k,v in self.other_mem_cost.items()}
            for k,v in self.other_mem_cost.items():
                for i in range(self.strategy_num):
                    if self.strategy_set[i][1]==k:
                        time_cost = sum(self.intra_cost[:,i]) + sum(self.inter_cost[:,i,i]) + self.other_time_cost[k]
                        mem_cost = sum(self.v_data[:,i]) + self.other_mem_cost[k]
                        if self.max_mem - 1 - mem_cost >= 0 and total_cost[k] > time_cost:
                            remaining_mem[k] = self.max_mem - 1 - mem_cost
                            total_cost[k] = time_cost
                            res_list[k] = np.full((self.layer_num), i, dtype=np.int32)
            return total_cost, res_list, remaining_mem 
        else:
            assert self.use_cpp_core == True    
            if self.use_cpp_core:     
                current_script_path = os.path.abspath(__file__)
                current_script_dir = os.path.dirname(current_script_path)
                sys.path.append(current_script_dir)
                # print(f'already add {current_script_dir} to sys')
                import galvatron_dp_core
                res_list = {k:np.full((self.layer_num), -1, dtype=np.int32) for k,v in self.other_mem_cost.items()}
                total_cost, remaining_mem = galvatron_dp_core.dynamic_programming_core(
                    self.layer_num, 
                    self.max_mem, 
                    self.strategy_num, 
                    self.v_data, 
                    self._mark, 
                    self._f, 
                    self.inter_cost, 
                    self.intra_cost,
                    self.other_mem_cost,
                    self.other_time_cost,
                    res_list,
                    )
                res_list = {k:list(v) for k,v in res_list.items()}

                return total_cost, res_list, remaining_mem

class DpOnModel:
    def __init__(self, 
                 strategies_set:List[LayerWiseStrategy], 
                 world_size, 
                 pp_stage_dict,
                 memory_cost_model_args_dict=None,
                 other_memory_cost_model_args_dict=None,
                 time_cost_model_args_dict=None,
                 other_time_cost_args_dict=None,
                 fine_grained=False,
                 layer_num=8,
                 max_mem=24, 
                 mem_cache_flag=True, 
                 sequence_length=1024,
                 hidden_size=4096,
                 mixed_precision='bf16',
                 allreduce_coe_dict=None,
                 logger=None):
        
        assert isinstance(layer_num, int), f'layer_num should be an integer, but got {type(layer_num)}' #  Decoder-encoder models not supported yet
        assert memory_cost_model_args_dict is not None
        assert other_memory_cost_model_args_dict is not None
        assert time_cost_model_args_dict is not None
        assert other_time_cost_args_dict is not None
        assert allreduce_coe_dict is not None
        
        self.strategies_set = strategies_set
        self.pp_deg_set = sorted(list(set([strategy.pp_size for strategy in strategies_set])))
        self.world_size = world_size
        self.layer_num = layer_num
        self.logger = logger
        self.pp_stage_dict = pp_stage_dict
        self.fine_grained = fine_grained
        self.memory_cost_model_args_dict = memory_cost_model_args_dict
        self.other_memory_cost_model_args_dict = other_memory_cost_model_args_dict
        self.time_cost_model_args_dicts = time_cost_model_args_dict
        self.other_time_cost_args_dict = other_time_cost_args_dict
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.mixed_precision = mixed_precision
        self.allreduce_coe_dict = allreduce_coe_dict
        
        # memory constraints
        self.max_mem = max_mem #  in MB
        self.mem_cache = 0
        if self.max_mem // 1024 > 20 and mem_cache_flag:
            mem_cache_ratio = 0.30
            self.mem_cache = int(self.max_mem * mem_cache_ratio) # reserved memory for paddle memory cache
            self.max_mem -= self.mem_cache
    
    def match_strategy(self, s1:LayerWiseStrategy, s2:LayerWiseStrategy, except_keys=[]):
        if s1.pp_size != s2.pp_size or s1.tp_size != s2.tp_size or s1.dp_size != s2.dp_size:
            return False
        check_equal_keys = ['ulysses', 'recompute', 'sharding_stage']
        for key in check_equal_keys:
            if key in except_keys: # skip the keys in except_keys
                continue
            if key == 'ulysses' and s1.use_ulysses != s2.use_ulysses:
                return False
            if key == 'recompute' and s1.recompute != s2.recompute:
                return False
            if key == 'sharding_stage' and s1.sharding_stage != s2.sharding_stage:
                return False
        return True
    
    def _build_dp_and_run_multi_layer_type(self, pp_deg, bsz, mbsz, min_tp, max_tp, vsp, embed_sdp, sp_search): # mbsz means when pp=pp_deg and tp=min_tp
        # calculate accumulation_steps and filter strategy_set
        accumulation_steps = bsz // (self.world_size // (pp_deg * min_tp) * mbsz)
        strategy_set = list(filter(lambda s: s.pp_size == pp_deg, self.strategies_set))
        strategy_num = len(strategy_set)
        
        # [Step1] calculate intra_layer_cost
        intra_layer_cost = []
        for strategy in strategy_set:
            time_cost_model_args = TimeCostModelArguments(strategy=strategy, global_batch_size=bsz // accumulation_steps, **self.time_cost_model_args_dicts)
            intra_layer_cost.append(TimeCostModel(time_cost_model_args).gen_result())        
        for i, strategy in enumerate(strategy_set):
            self.logger.info(f'{i}-th, strategy:{strategy}, time_cost: {intra_layer_cost[i]}')
        
        intra_layer_cost = np.array(intra_layer_cost, dtype=np.float64).reshape(1, -1).repeat(self.layer_num, axis=0)
        min_cost_strategy_ids = np.argmin(intra_layer_cost, axis=1)
        self.logger.info(f'min_cost_strategy_id: {min_cost_strategy_ids[0]}')
        
        # [Step2] calculate othertimecost
        other_time_cost_args = OtherTimeCostModelArguments(pp_size=pp_deg, min_tp_size=min_tp, max_tp_size=max_tp, embed_sdp=embed_sdp, micro_batch_size=mbsz, vocab_use_ulysees=vsp, **self.other_time_cost_args_dict)
        other_time_cost = OtherTimeCostModel(other_time_cost_args).gen_result()
        self.logger.info(f'other_time_cost: {other_time_cost}')
        
        # [Step3] calculate inter cost
        inter_layer_cost = np.zeros((strategy_num, strategy_num))
        # calculate communication size
        for i in range(strategy_num):
            for j in range(strategy_num):
                case1 = strategy_set[i].tp_size > strategy_set[j].tp_size #  tp down, dp up, bsz down
                case2 = False
                case3 = False
                # if strategy_set[i].tp_size != 1 and strategy_set[j].tp_size != 1:
                sample_num = self.sequence_length * self.hidden_size * (4 if self.mixed_precision == 'fp32' else 2) #  sequence_length is not defined, should be passed as an argument
                # sample_num = self.sequence_len[idx] * self.config.hidden_size * (4 if self.config.mixed_precision == "fp32" else 2)
                case4  = strategy_set[i].tp_size !=  strategy_set[j].tp_size
                if case1 or case2 or case3 or case4:
                    nw_max_tp = max(strategy_set[i].tp_size, strategy_set[j].tp_size)
                    inter_layer_cost[i,  j] = (nw_max_tp - 1) / nw_max_tp * mbsz * (nw_max_tp // min_tp) * sample_num
        # calculate communication cost
        for i in range(strategy_num):
            for j in range(strategy_num):
                tp_size = max(strategy_set[i].tp_size, strategy_set[j].tp_size)
                # dp_size = min(strategy_set[i].dp_size, strategy_set[j].dp_size)
                coe = self.allreduce_coe_dict[tp_size]
                inter_layer_cost[i, j] = inter_layer_cost[i, j] * coe * 1e-7 # NOTE times 1e-7
                
                # add a small bias to sort fsdp and dp
                strategy0, strategy1 = strategy_set[i], strategy_set[j]
                # tp -> sp
                if i != j and self.match_strategy(strategy0, strategy1, except_keys=['ulysses']):
                    if strategy1.use_ulysses:
                        inter_layer_cost[i, j] = 1e-10
                # ->f     c -> fc
                if i != j and self.match_strategy(strategy0, strategy1, except_keys=['sharding_stage']):
                    if strategy1.sharding_stage == 3:
                        inter_layer_cost[i, j] = 1e-9
                # ->c  f -> cf
                if i != j and self.match_strategy(strategy0, strategy1, except_keys=['recompute']):
                    if strategy1.recompute:
                        inter_layer_cost[i, j] = 2e-9
                # ->fc
                if i != j and self.match_strategy(strategy0, strategy1, except_keys=['sharding_stage','recompute']):
                    if strategy1.sharding_stage == 3 and strategy1.recompute:
                        inter_layer_cost[i, j] = 3e-9
                # f->c
                if i != j and self.match_strategy(strategy0, strategy1, except_keys=['sharding_stage','recompute']) \
                        and not self.match_strategy(strategy0, strategy1, except_keys=['sharding_stage']) \
                        and not self.match_strategy(strategy0, strategy1, except_keys=['recompute']):
                    if strategy0.sharding_stage == 3 and strategy1.recompute:
                        inter_layer_cost[i, j] = 1e-9
        
        inter_layer_cost = np.expand_dims(inter_layer_cost, axis=0).repeat(self.layer_num, axis=0)
        inter_layer_cost[0, :, :] = 0 # no inter-layer communication cost in first layer
        
        # calculate memory cost
        v_list_stage_idx = []
        for stage_idx in range(pp_deg):
            mem_cost_list = []
            for i, strategy in enumerate(strategy_set):
                memory_cost_model_agrs = MemoryCostModelArguments(strategy=strategy, global_batch_size=bsz, stage_idx=stage_idx, accumulation_steps=accumulation_steps, **self.memory_cost_model_args_dict)
                mem_cost_list.append(MemoryCostModel(memory_cost_model_agrs).get_memory_cost())
                
            v = [cost['enc_total'] for cost in mem_cost_list]
            v = np.ceil(np.array(v)).astype(np.int32) # convert to int32
            v = v.reshape(1, -1).repeat(self.layer_num, axis=0)
            v_list_stage_idx.append(v)
        for stage_idx in range(pp_deg):
            self.logger.info(f'stage_idx{stage_idx}, memory_cost:{v_list_stage_idx[stage_idx][0]}')

        # calculate other memory cost
        # TODO sharding_stage需要进行修改
        other_mem_cost_args = OtherMemoryCostModelArguments(min_tp_size=min_tp, max_tp_size=max_tp, pp_size=pp_deg, sharding_stage=3, global_batch_size=bsz, accumulation_steps=accumulation_steps, use_ulysses=vsp, **self.other_memory_cost_model_args_dict)
        other_mem_cost = OtherMemoryCostModel(other_mem_cost_args).get_other_memory_cost()
        for key, value in other_mem_cost.items():
            other_mem_cost[key] = np.ceil(value).astype(int)
        self.logger.info(f'other_mem_cost: {other_mem_cost}')

        # start solve
        pp_stage_list = self.pp_stage_dict[pp_deg]
        start_layer = 0
        comm_cost_list, res_list_list, mem_remain_list, mem_cost_list = [], [], [], []
        best_strategy_flag = {k:[False for i in range(pp_deg)] for k,v in other_mem_cost.items()} # TODO 没有看懂这个
        from_history = None

        if self.fine_grained == False:
            print(f'[linguangming] not use fine_grained')
            pass
        else: # fine-grained search
            # DP solution for each stage
            for stage_idx in range(pp_deg):
                global_memory = 0
                
                # select other_mem_cost and other_time_cost for this stage
                nw_other_mem_cost = {k:v[stage_idx] + int(global_memory) for k,v in other_mem_cost.items()}
                nw_other_time_cost = {k:v[stage_idx] for k,v in other_time_cost[0].items()}
                mem_cost = {k:0 for k,v in other_time_cost[0].items()}
                
                # select memory_cost for this stage
                v = v_list_stage_idx[stage_idx] # v.shape = (self.layer_num, strategy_num)

                dp = DPAlg(max_mem=self.max_mem, 
                           other_mem_cost=nw_other_mem_cost, 
                           other_time_cost=nw_other_time_cost, 
                           layer_num=pp_stage_list[stage_idx], 
                           strategy_num=strategy_num,
                           strategy_set=strategy_set,)
                dp.set_v_and_cost(v[start_layer:start_layer + pp_stage_list[stage_idx]], # select layers for this stage
                                  intra_layer_cost[start_layer:start_layer + pp_stage_list[stage_idx]], #  select layers for this stage
                                  inter_layer_cost[start_layer:start_layer + pp_stage_list[stage_idx]]) # select layers for this stage
                comm_cost, res_list, mem_remain = dp.fit()
                # self.logger.info(f'[linguangming] stage_idx:{stage_idx}, mem_remain:{mem_remain}')
                
                for k, v in comm_cost.items():
                    if mem_remain[k] == -1:
                        res_list[k] = None

                    best_strategy_flag[k][stage_idx] = res_list[k] is not None and (np.array(res_list[k]) == min_cost_strategy_ids[start_layer:start_layer+pp_stage_list[stage_idx]]).all()
                    if res_list[k] is not None:
                        res_list[k] = list(map(lambda x: strategy_set[x], res_list[k]))
                    mem_cost[k] = self.max_mem - mem_remain[k] if mem_remain[k] >= 0 else np.inf
                    
                comm_cost_list.append(comm_cost)
                res_list_list.append(res_list)
                mem_remain_list.append(mem_remain)
                mem_cost_list.append(mem_cost)
                start_layer += pp_stage_list[stage_idx]
                
            # search the best vtp
            comm_cost = np.inf
            vtp = -1
            for k in other_time_cost[0].keys():
                nw_res_list_list = [v2[k] for v2 in res_list_list]
                # nw_comm_cost_list = [v2[k] for v2 in comm_cost_list]
                if None not in nw_res_list_list:
                    res_list = []
                    for res in nw_res_list_list:
                        res_list += res
                    pipeline_cost = pipeline_costmodel(self.time_cost_model_args_dicts, res_list=res_list, accumulation_steps=accumulation_steps, global_batch_size=bsz, pp_stage_list=pp_stage_list, other_time_cost_no_comm=other_time_cost[1][k])
                    self.logger.info(f'[linguangming] pipeline cost: {pipeline_cost}')
                    if comm_cost > pipeline_cost:
                        comm_cost = pipeline_cost
                        vtp = k
            if vtp != -1:
                res_list_list = [v2[vtp] for v2 in res_list_list]
                mem_remain_list = [v2[vtp] for v2 in mem_remain_list]
                mem_cost_list = [v2[vtp] for v2 in mem_cost_list]
            else:
                res_list_list, mem_remain_list, mem_cost_list = None, [-1 for v2 in mem_remain_list], [-1 for v2 in mem_cost_list]
            
            # print(f'[linguangming-test] comm_cost:{comm_cost}, res_list_list:{res_list_list}, mem_remain_list:{mem_remain_list}, mem_cost_list:{mem_cost_list}')
            return comm_cost, res_list_list, mem_remain_list, mem_cost_list, vtp, best_strategy_flag, from_history

    def fit(self, bsz, min_tp, max_tp, vsp, embed_sdp, sp_search, print_=True, mbsz_dict=None):
        optimal_comm_cost = np.inf
        optimal_res_list = None
        optimal_pp_deg = -1
        optimal_mem_remain = -1
        optimal_mem_cost = -1
        optimal_vtp = -1

        for pp_deg in self.pp_deg_set:
            # Actually, this should never happen since DpOnModel initialization already ensures correctness
            if pp_deg * min_tp > self.world_size:
                continue # Skip this pp_deg
            
            if print_:
                if self.logger is not None:
                    self.logger.info(f'pp_deg={pp_deg} bsz={bsz} min_tp={min_tp} max_tp={max_tp} vsp={vsp} embed_sdp={embed_sdp}, sp_search={sp_search}')
                else:
                    print(f'pp_deg={pp_deg} bsz={bsz} min_tp={min_tp} max_tp={max_tp} vsp={vsp} embed_sdp={embed_sdp}, sp_search={sp_search}')
            
            # Actually, this should never happen since DpOnModel initialization already ensures correctness
            if bsz % (self.world_size // (pp_deg * min_tp)):
                comm_cost, res_list, mem_remain, mem_cost, best_strategy_flag, from_history = np.inf, None, -1, np.inf, False, False
                if optimal_res_list is None:
                    optimal_res_list = '[current bsz is not divisible by bsz_scale]'
                if print_:
                    if self.logger is not None:
                        self.logger.info(f'Best strategy: {best_strategy_flag} \nFrom history: {from_history}')
                        self.logger.info(f'time cost: {comm_cost}, memory remaining: {mem_remain}, memory cost: {mem_cost}')
                    else:
                        print(f'Best strategy: {best_strategy_flag} \nFrom history: {from_history}')
                        print(f'time cost: {comm_cost}, memory remaining: {mem_remain}, memory cost: {mem_cost}')
                continue # Skip this pp_deg
            
            # pp_deg is valid, proceed with dynamic programming
            comm_cost, res_list, mem_remain, mem_cost, vtp, best_strategy_flag, from_history = self._build_dp_and_run_multi_layer_type(pp_deg, bsz, mbsz_dict[pp_deg], min_tp, max_tp, vsp, embed_sdp, sp_search)
            self.logger.info(f'[linguangming] [debug] origin mem_remain: {mem_remain}, origin mem_cost: {mem_cost}')
            mem_cost = [m + self.mem_cache for m in mem_cost] if isinstance(mem_cost, list) else mem_cost + self.mem_cache 
            
            if print_:
                if self.logger is not None:
                    # self.logger.info(f'Best strategy: {best_strategy_flag} \nFrom history: {from_history}')
                    self.logger.info(f'time cost: {comm_cost}, memory remaining: {mem_remain}, memory cost: {mem_cost}')
                else:
                    # print(f'Best strategy: {best_strategy_flag} \nFrom history: {from_history}')
                    print(f'time cost: {comm_cost}, memory remaining: {mem_remain}, memory cost: {mem_cost}')   
            if optimal_comm_cost > comm_cost:
                optimal_res_list = res_list
                optimal_comm_cost = comm_cost
                optimal_pp_deg = pp_deg
                optimal_mem_remain = mem_remain
                optimal_mem_cost = mem_cost
                optimal_vtp = vtp

        return optimal_comm_cost, optimal_res_list, optimal_pp_deg, optimal_mem_remain, optimal_mem_cost, optimal_vtp
        