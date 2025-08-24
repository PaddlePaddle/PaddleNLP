from paddlenlp.experimental.galvatron.cost_model.profile_data_parser import ProfileDataParser, ProfileDataParserArguments
from paddlenlp.experimental.galvatron.utils import get_current_all_args, Strategy, LayerWiseStrategy

cases = [ 
            {
                'strategy_dict': {'pp_size': 1, 'tp_size': 4, 'use_ulysses': 0, 'dp_size':2, 'sharding_stage': 2, 'recompute': 0}, 
                'global_batch_size': 32, 
                'accumulation_steps': 8
            },
            {
                'strategy_dict': {'pp_size': 1, 'tp_size': 2, 'use_ulysses': 0, 'dp_size':4, 'sharding_stage': 2, 'recompute': 0}, 
                'global_batch_size': 32, 
                'accumulation_steps': 4
            },
            # {
            #     'strategy_dict': {'pp_size': 1, 'tp_size': 1, 'use_ulysses': 0, 'dp_size':8, 'sharding_stage': 2, 'recompute': 0}, 
            #     'global_batch_size': 32, 
            #     'accumulation_steps': 2
            # },
            # {
            #     'strategy_dict': {'pp_size': 1, 'tp_size': 8, 'use_ulysses': 0, 'dp_size':1, 'sharding_stage': 2, 'recompute': 0}, 
            #     'global_batch_size': 32, 
            #     'accumulation_steps': 8
            # },
            # {
            #     'strategy_dict': {'pp_size': 1, 'tp_size': 4, 'use_ulysses': 0, 'dp_size':2, 'sharding_stage': 2, 'recompute': 1}, 
            #     'global_batch_size': 32, 
            #     'accumulation_steps': 8
            # },
            # {
            #     'strategy_dict': {'pp_size': 1, 'tp_size': 2, 'use_ulysses': 0, 'dp_size':4, 'sharding_stage': 2, 'recompute': 1}, 
            #     'global_batch_size': 32, 
            #     'accumulation_steps': 4
            # },
            # {
            #     'strategy_dict': {'pp_size': 1, 'tp_size': 4, 'use_ulysses': 0, 'dp_size':2, 'sharding_stage': 3, 'recompute': 0}, 
            #     'global_batch_size': 32, 
            #     'accumulation_steps': 8
            # },
            # {
            #     'strategy_dict': {'pp_size': 1, 'tp_size': 2, 'use_ulysses': 0, 'dp_size':4, 'sharding_stage': 3, 'recompute': 0}, 
            #     'global_batch_size': 32, 
            #     'accumulation_steps': 4
            # },
            # {
            #     'strategy_dict': {'pp_size': 1, 'tp_size': 4, 'use_ulysses': 1, 'dp_size':2, 'sharding_stage': 2, 'recompute': 0}, 
            #     'global_batch_size': 32, 
            #     'accumulation_steps': 8
            # },
            # {
            #     'strategy_dict': {'pp_size': 1, 'tp_size': 2, 'use_ulysses': 1, 'dp_size':4, 'sharding_stage': 2, 'recompute': 0}, 
            #     'global_batch_size': 32, 
            #     'accumulation_steps': 4
            # },
            # {
            #     'strategy_dict': {'pp_size': 2, 'tp_size': 2, 'use_ulysses': 0, 'dp_size':2, 'sharding_stage': 2, 'recompute': 0}, 
            #     'global_batch_size': 32, 
            #     'accumulation_steps': 4
            # },
            # {
            #     'strategy_dict': {'pp_size': 4, 'tp_size': 2, 'use_ulysses': 0, 'dp_size':1, 'sharding_stage': 2, 'recompute': 0}, 
            #     'global_batch_size': 32, 
            #     'accumulation_steps': 8
            # },
            # {
            #     'strategy_dict': {'pp_size': 4, 'tp_size': 1, 'use_ulysses': 0, 'dp_size':2, 'sharding_stage': 2, 'recompute': 0}, 
            #     'global_batch_size': 32, 
            #     'accumulation_steps': 8
            # },
            # {
            #     'strategy_dict': {'pp_size': 1, 'tp_size': 2, 'use_ulysses': 0, 'dp_size':4, 'sharding_stage': 2, 'recompute': 0}, 
            #     'global_batch_size': 16, 
            #     'accumulation_steps': 1
            # },
        ]

def get_result(case):
    strategy_dict = case['strategy_dict']
    if strategy_dict['dp_size'] == 1:
        strategy_dict['sharding_stage'] = 0
    strategy = LayerWiseStrategy.deserialize(strategy_dict)

    global_batch_size = case['global_batch_size']
    mixed_precision_type = 'bf16'
    accumulation_steps = case['accumulation_steps']
    
    print(f'\nmemory cost calculating...')
    memory_cost = profile_data_parser.get_memory_cost_for_specific_strategy(strategy, global_batch_size, mixed_precision_type, accumulation_steps)
    print('======== memory cost result ========')
    for stage_idx in range(strategy.pp_size):
        print(f'stage {stage_idx}: {memory_cost[stage_idx]} MB, {memory_cost[stage_idx] / 1024} GB')
        
    print(f'\ntime cost calculating...')
    time_cost = profile_data_parser.get_time_cost_for_specific_strategy(strategy, global_batch_size, mixed_precision_type, accumulation_steps)
    print('======== time cost result ========')
    print(f'time cost: {time_cost}')

    return memory_cost, time_cost


if __name__ == "__main__":
    args_dict = get_current_all_args()
    
    profile_data_parser_args = ProfileDataParserArguments()
    profile_data_parser_args.initialize(args_dict=args_dict)
    profile_data_parser = ProfileDataParser(profile_data_parser_args)
    print('profile_data_parser constructed.')
    
    results = []
    for case in cases:
        res = get_result(case)
        results.append(res)
        print('\n\n\n')
    
    
    print(f'最后汇总')
    for res in results:
        memory_cost, time_cost = res
        print(f'memory cost: {memory_cost}, time cost: {time_cost}')

    
    