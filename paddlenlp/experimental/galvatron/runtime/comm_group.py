from paddle.distributed.auto_parallel.process_mesh import ProcessMesh
import json

def get_pp_division_ranks(gpu_nums, pp_deg):
    assert gpu_nums % pp_deg == 0, f'gpu_nums {gpu_nums} should be divisible by pp_deg {pp_deg}'
    result = []
    for i in range(pp_deg):
        pp_group = []
        for j in range(gpu_nums // pp_deg):
            pp_group.append(i * (gpu_nums // pp_deg) + j)
        result.append(pp_group)
    return result

def get_dp_tp_sp_ranks(rank_list, dp_deg, tp_deg, sep_deg):
    if tp_deg > 1 and sep_deg > 1:
        raise ValueError("tp_deg and sep_deg cannot be greater than 1 at the same time.")
    assert len(rank_list) == dp_deg * tp_deg or len(rank_list) == dp_deg * sep_deg, f'len(rank_list) {len(rank_list)} should equal dp_deg * tp_deg {dp_deg * tp_deg} or dp_deg * sep_deg {dp_deg * sep_deg}'
    
    result = []
    for i in range(dp_deg):
        dp_group = []
        for j in range(max(tp_deg, sep_deg)):
            dp_group.append(rank_list[i * max(tp_deg, sep_deg) + j])
        result.append(dp_group)
        
    return result

def get_comm_meshs(gpu_nums, pp_deg, dp_deg_list, tp_deg_list, sep_deg_list, pp_stage_idx_list, use_tensor_parallel=True):
    assert len(dp_deg_list) == len(tp_deg_list) and len(dp_deg_list) == len(sep_deg_list), "dp_deg_list, tp_deg_list and sep_deg_list should have the same length."
    assert len(dp_deg_list) == len(pp_stage_idx_list), "dp_deg_list and pp_stage_idx_list should have the same length."
    assert pp_deg > 0, "pp_deg should be greater than 0."
    assert gpu_nums % pp_deg == 0, f'gpu_nums {gpu_nums} should be divisible by pp_deg {pp_deg}'
    
    for dp_deg, tp_deg, sep_deg in zip(dp_deg_list, tp_deg_list, sep_deg_list):
        if tp_deg > 1 and sep_deg > 1:
            raise ValueError("tp_deg and sep_deg cannot be greater than 1 at the same time.")
        assert gpu_nums == pp_deg * dp_deg * tp_deg or gpu_nums == pp_deg * dp_deg * sep_deg, f'gpu_nums {gpu_nums} should equal pp_deg * dp_deg * tp_deg {pp_deg * dp_deg * tp_deg} or pp_deg * dp_deg * sep_deg {pp_deg * dp_deg * sep_deg}'
    
    pp_ranks_division = get_pp_division_ranks(gpu_nums, pp_deg)
    
    comm_meshs = []
    for dp_deg, tp_deg, sep_deg, pp_stage in zip(dp_deg_list, tp_deg_list, sep_deg_list, pp_stage_idx_list):
        result = get_dp_tp_sp_ranks(pp_ranks_division[pp_stage], dp_deg, tp_deg, sep_deg)
        mesh = ProcessMesh(result, dim_names=["dp", "tp" if use_tensor_parallel else "sep"])
        comm_meshs.append(mesh)
        
    return comm_meshs

def get_comm_meshs_coarse_grained(gpu_nums, pp_deg, dp_deg, tp_deg, sep_deg, num_hidden_layers, use_tensor_parallel=True):
    assert gpu_nums % pp_deg == 0, f'gpu_nums {gpu_nums} should be divisible by pp_deg {pp_deg}'
    if tp_deg > 1 and sep_deg > 1:
        raise ValueError("tp_deg and sep_deg cannot be greater than 1 at the same time.")
    assert gpu_nums == pp_deg * dp_deg * tp_deg or gpu_nums == pp_deg * dp_deg * sep_deg, f'gpu_nums {gpu_nums} should equal pp_deg * dp_deg * tp_deg {pp_deg * dp_deg * tp_deg} or pp_deg * dp_deg * sep_deg {pp_deg * dp_deg * sep_deg}'
    
    print(f'[comm_group.py] init comm meshs coarse-grained')
    
    dp_deg_list = [dp_deg] * (1 + num_hidden_layers + 2)
    tp_deg_list = [tp_deg] * (1 + num_hidden_layers + 2)
    sep_deg_list = [sep_deg] * (1 + num_hidden_layers + 2)
    
    if num_hidden_layers % pp_deg == 0:
        per_stage_num_layers = num_hidden_layers // pp_deg
        pp_stage_idx_list = [0] + [i // per_stage_num_layers for i in range(0, num_hidden_layers)] + [pp_deg - 1, pp_deg - 1]
    else:
        per_stage_num_layers = num_hidden_layers
        remain_num_layers = num_hidden_layers % pp_deg
        pp_stage_idx_list = [0] + [i // per_stage_num_layers for i in range(0, num_hidden_layers - remain_num_layers)] + [pp_deg - 1] * remain_num_layers + [pp_deg - 1, pp_deg - 1]
    
    print(f'[comm_group.py] pp_stage_idx_list: {pp_stage_idx_list}')
    print(f'[comm_group.py] dp_deg_list: {dp_deg_list}')
    print(f'[comm_group.py] tp_deg_list: {tp_deg_list}')
    print(f'[comm_group.py] sep_deg_list: {sep_deg_list}')
    
    return get_comm_meshs(gpu_nums, pp_deg, dp_deg_list, tp_deg_list, sep_deg_list, pp_stage_idx_list, use_tensor_parallel)

def get_comm_meshs_fine_grained(gpu_nums, config_file_path):
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    
    pp_deg = int(config['pp_deg'])
    dp_deg_list = [int(i) for i in config['dp_deg_list'].split(',')]
    tp_deg_list = [int(i) for i in config['tp_deg_list'].split(',')]
    sep_deg_list = [int(i) for i in config['sep_deg_list'].split(',')]
    pp_stage_idx_list = [int(i) for i in config['pp_stage_idx_list'].split(',')]
    use_tensor_parallel = int(config['use_tensor_parallel'])
    
    return get_comm_meshs(gpu_nums, pp_deg, dp_deg_list, tp_deg_list, sep_deg_list, pp_stage_idx_list, use_tensor_parallel)
        
def get_redistributed_flag(mesh_list, rank, use_tensor_parallel=True):
    flag = [0] * len(mesh_list)
    for i, mesh in enumerate(mesh_list):
        # if rank not in mesh.process_ids:
        #     continue
        if i != len(mesh_list) - 1:
            next_mesh_dp_deg = mesh_list[i + 1].shape[0]
            now_hidden_states_dp_deg = mesh.shape[0]
            if next_mesh_dp_deg != now_hidden_states_dp_deg:
                shape = get_dp_tp_sp_ranks(mesh.process_ids, next_mesh_dp_deg, mesh_list[i + 1].shape[1], 1) if use_tensor_parallel else get_dp_tp_sp_ranks(mesh.process_ids, next_mesh_dp_deg, 1, mesh_list[i + 1].shape[1])
                to_mesh = ProcessMesh(shape, dim_names=["dp", "tp" if use_tensor_parallel else "sep"])
                flag[i] = to_mesh
    return flag
        
        
    