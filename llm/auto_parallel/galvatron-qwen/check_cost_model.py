from paddlenlp.experimental.galvatron.cost_model.profile_data_parser import ProfileDataParser, ProfileDataParserArguments
from paddlenlp.experimental.galvatron.utils import get_current_all_args, Strategy, LayerWiseStrategy

if __name__ == "__main__":
    args_dict = get_current_all_args()
    
    profile_data_parser_args = ProfileDataParserArguments()
    profile_data_parser_args.initialize(args_dict=args_dict)
    profile_data_parser = ProfileDataParser(profile_data_parser_args)
    print('profile_data_parser constructed.')
    
    # strategy_str = args_dict.pop("--strategy", None)
    global_batch_size = int(args_dict.pop("--global_batch_size", 1))
    mixed_precision_type = args_dict.pop("--mixed_precision_type", False)
    accumulation_steps = int(args_dict.pop("--accumulation_steps", 1))
    
    # assert strategy_str is not None, "Strategy must be specified."
    # strategy = Strategy()
    # strategy.deserialize(strategy_str)
    # strategy = LayerWiseStrategy.deserialize(strategy_str)
    
    
    # 以下先进行硬编码
    strategy_dict = {
        'pp_size': 4,
        'tp_size': 1,
        'use_ulysses': 0,
        'dp_size': 2,
        'sharding_stage': 2,
        'recompute': 1
    }
    global_batch_size = 32
    accumulation_steps = 8
    
    if strategy_dict['dp_size'] == 1:
        strategy_dict['sharding_stage'] = 0
    
    strategy = LayerWiseStrategy.deserialize(strategy_dict)
    
    print(f'current strategy: {strategy}')
    
    print(f'\nmemory cost calculating...')
    memory_cost = profile_data_parser.get_memory_cost_for_specific_strategy(strategy, global_batch_size, mixed_precision_type, accumulation_steps)
    print('======== memory cost result ========')
    for stage_idx in range(strategy.pp_size):
        print(f'stage {stage_idx}: {memory_cost[stage_idx]} MB, {memory_cost[stage_idx] / 1024} GB')
        
    # print(f'\ntime cost calculating...')
    # time_cost = profile_data_parser.get_time_cost_for_specific_strategy(strategy, global_batch_size, mixed_precision_type, accumulation_steps)
    # print('======== time cost result ========')
    # print(f'time cost: {time_cost}')
    
    