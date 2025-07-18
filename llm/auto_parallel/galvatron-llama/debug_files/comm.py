from paddlenlp.experimental.galvatron.runtime.comm_group import get_pp_division_ranks, get_dp_tp_sp_ranks, get_comm_meshs, get_comm_meshs_coarse_grained, get_comm_meshs_fine_grained, get_redistributed_flag
import random
import json

def comm_test(gpu_nums, pp_deg, dp_deg, tp_deg, sep_deg):
    print(f'comm_test: gpu_nums {gpu_nums}, pp_deg {pp_deg}, dp_deg {dp_deg}, tp_deg {tp_deg}, sep_deg {sep_deg}')
    pp_ranks_division = get_pp_division_ranks(gpu_nums, pp_deg)
    print(f'pp_ranks_division: {pp_ranks_division}')
    for i, pp_rank_list in enumerate(pp_ranks_division):
        result = get_dp_tp_sp_ranks(pp_rank_list, dp_deg, tp_deg, sep_deg)
        print(f'stage {i}: {result}')
        
def mesh_test(gpu_nums, pp_deg, num_layers):
    print(f'mesh_test: gpu_nums {gpu_nums}, pp_deg {pp_deg}, num_layers {num_layers}')
    max_deg = gpu_nums // pp_deg
    
    i = 1
    select_deg = []
    while i <= max_deg:
        select_deg.append(i)
        i *= 2
    
    dp_deg_list, tp_deg_list, sep_deg_list = [], [], []
    pp_stage_idx_list = []
    
    # embedding mesh
    dp_deg = select_deg[random.randint(0, len(select_deg) - 1)]
    tp_deg = max_deg // dp_deg
    sep_deg = 1
    dp_deg_list.append(dp_deg)
    tp_deg_list.append(tp_deg)
    sep_deg_list.append(sep_deg)
    
    # transformer layers mesh
    for i in range(num_layers):
        dp_deg = select_deg[random.randint(0, len(select_deg) - 1)]
        tp_deg = max_deg // dp_deg
        sep_deg = 1
        dp_deg_list.append(dp_deg)
        tp_deg_list.append(tp_deg)
        sep_deg_list.append(sep_deg)
        
    # lm and loss
    dp_deg = select_deg[random.randint(0, len(select_deg) - 1)]
    tp_deg = max_deg // dp_deg
    sep_deg = 1
    dp_deg_list.append(dp_deg)
    tp_deg_list.append(tp_deg)
    sep_deg_list.append(sep_deg)
    dp_deg = select_deg[random.randint(0, len(select_deg) - 1)]
    tp_deg = max_deg // dp_deg
    sep_deg = 1
    dp_deg_list.append(dp_deg)
    tp_deg_list.append(tp_deg)
    sep_deg_list.append(sep_deg)
    
    # pp stage idx
    pp_stage_idx_list.append(0) # embedding
    layer_num_per_stage = num_layers // pp_deg
    for i in range(pp_deg):
        pp_stage_idx_list.extend([i] * layer_num_per_stage)
    pp_stage_idx_list.append(pp_deg - 1) # lm and loss
    pp_stage_idx_list.append(pp_deg - 1) # last layer
    
    print(f'dp_deg_list: {dp_deg_list}')
    print(f'tp_deg_list: {tp_deg_list}')
    print(f'sep_deg_list: {sep_deg_list}')
    print(f'pp_stage_idx_list: {pp_stage_idx_list}')
    
    comm_meshs = get_comm_meshs(gpu_nums, pp_deg, dp_deg_list, tp_deg_list, sep_deg_list, pp_stage_idx_list)
    for i, mesh in enumerate(comm_meshs):
        print(f'mesh {i}: {mesh}')

def comm_test_coarse_grained(gpu_nums, pp_deg, dp_deg, tp_deg, sep_deg, num_hidden_layers):
    print(f'comm_test_coarse_grained: gpu_nums {gpu_nums}, pp_deg {pp_deg}, dp_deg {dp_deg}, tp_deg {tp_deg}, sep_deg {sep_deg}, num_hidden_layers {num_hidden_layers}')
    comm_meshs = get_comm_meshs_coarse_grained(gpu_nums, pp_deg, dp_deg, tp_deg, sep_deg, num_hidden_layers)
    for i, mesh in enumerate(comm_meshs):
        print(f'mesh {i}: {mesh}')

def comm_test_fine_grained(gpu_nums):
    # manually set config
    config = {}
    config['pp_deg'] = 2
    num_hidden_layers = 8
    config['use_tensor_parallel'] = 1
    
    config['dp_deg_list'] = [2] * (1 + num_hidden_layers + 2)
    config['tp_deg_list'] = [2] * (1 + num_hidden_layers + 2)
    config['sep_deg_list'] = [1] * (1 + num_hidden_layers + 2)
    config['pp_stage_idx_list'] = [0] + [i // (num_hidden_layers // config['pp_deg']) for i in range(0, num_hidden_layers)] + [config['pp_deg'] - 1, config['pp_deg'] - 1]
    
    modify = {
                2: {'dp_deg': 4, 'tp_deg':1},
                3: {'dp_deg': 1, 'tp_deg':4},
                5: {'dp_deg': 4, 'tp_deg':1},
                6: {'dp_deg': 1, 'tp_deg':4},
            }
    
    for key, value in modify.items():
        config['dp_deg_list'][key] = value['dp_deg']
        config['tp_deg_list'][key] = value['tp_deg']
    
    config['dp_deg_list'] = ",".join(map(str, config['dp_deg_list']))
    config['tp_deg_list'] = ",".join(map(str, config['tp_deg_list']))
    config['sep_deg_list'] = ",".join(map(str, config['sep_deg_list']))
    config['pp_stage_idx_list'] = ",".join(map(str, config['pp_stage_idx_list']))
    
    print(f'[comm_group.py] config: {config}')
    file_path = './config.json'
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    # set comm_meshs
    comm_meshs = get_comm_meshs_fine_grained(gpu_nums, file_path)
    for i, mesh in enumerate(comm_meshs):
        print(f'mesh {i}: {mesh}')
        
    # get redistributed flag
    for rank in range(0, 8):        
        flag = get_redistributed_flag(comm_meshs, rank, config['use_tensor_parallel'])
        print(f'[linguangming] for rank {rank}, flag: {flag}')
        
if __name__ == '__main__':
    # comm_test(8, 2, 2, 2, 1)
    # comm_test(8, 2, 2, 1, 2)
    # comm_test(8, 2, 4, 1, 1)
    # comm_test(8, 4, 2, 1, 1)
    # comm_test(8, 8, 1, 1, 1)
    
    # gpus_nums = 8
    # pp_deg = 2
    # dp_deg_list = [2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2]
    # tp_deg_list = [2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2]
    # sep_deg_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # pp_stage_idx_list = [0, 0, 0, 0, 0, 0, ]    
    # mesh_test(8, 2, 8)
    # comm_test_coarse_grained(8, 2, 2, 2, 1, 8)
    comm_test_fine_grained(8)