from paddle.distributed.auto_parallel.process_mesh import ProcessMesh
import json
import paddle.distributed as dist
from typing import List
from dataclasses import dataclass, field

@dataclass
class GranularityRuntimeArguments:
    granularity_type: str = field(default='coarse_grained', metadata={'help': 'Granularity type of the runtime, can be coarse_grained or fine_grained.'})
    usp_flag: int = field(default=0, metadata={'help': 'Ulysses sp'})
    fine_grained_config_path: str = field(default='', metadata={'help': 'Path to the fine-grained configuration file.'})
    sharding_stage_level: int = field(default=3, metadata={'help': 'Sharding stage level.'})

def get_pp_division_ranks(gpu_nums, pp_deg):
    assert gpu_nums % pp_deg == 0, f'gpu_nums {gpu_nums} should be divisible by pp_deg {pp_deg}'
    result = []
    for i in range(pp_deg):
        pp_group = []
        for j in range(gpu_nums // pp_deg):
            pp_group.append(i * (gpu_nums // pp_deg) + j)
        result.append(pp_group)
    return result

def get_dp_tp_ranks(rank_list, dp_size, tp_size):
    result = []
    for i in range(dp_size):
        dp_group = []
        for j in range(tp_size):
            dp_group.append(rank_list[i * tp_size + j])
        result.append(dp_group)
    return result

def generate_meshs(num_hidden_layers, gpu_nums, pp_size, pp_stage_idx_list, tp_size_list, usp_flag_list, dp_size_list, sharding_stage_list, recompute_list, vtp, vsp_flag):
    mesh_list:List[dist.ProcessMesh] = []
    pp_division_ranks = get_pp_division_ranks(gpu_nums, pp_size)
    print(pp_division_ranks)
    
    vocab_shape = get_dp_tp_ranks(pp_division_ranks[0], dp_size=gpu_nums // pp_size // vtp,  tp_size=vtp)
    print(vocab_shape)
    vocab_mesh = dist.ProcessMesh(vocab_shape, dim_names=['dp', 'tp' if vsp_flag == 0 else 'sep'])
    mesh_list.append(vocab_mesh)

    for tp_size, dp_size, usp_flag, stage_idx in zip(tp_size_list, dp_size_list, usp_flag_list, pp_stage_idx_list):
        res = get_dp_tp_ranks(pp_division_ranks[stage_idx], dp_size, tp_size)
        mesh = dist.ProcessMesh(res, dim_names=['dp', 'tp' if usp_flag == 0 else 'sep'])
        mesh_list.append(mesh)  
        
    cls_shape = get_dp_tp_ranks(pp_division_ranks[-1], dp_size=gpu_nums // pp_size  // vtp, tp_size=vtp)
    cls_mesh = dist.ProcessMesh(cls_shape, dim_names=['dp', 'tp' if vsp_flag == 0 else 'sep'])
    mesh_list.append(cls_mesh)

    assert len(mesh_list) == num_hidden_layers + 2, f'len(mesh_list) {len(mesh_list)} should equal num_hidden_layers {num_hidden_layers} + 2'

    for mesh in mesh_list:
        print(mesh)

    return mesh_list, pp_stage_idx_list, recompute_list, sharding_stage_list

def generate_meshs_coarse_grained(num_hidden_layers, gpu_nums, pp_size, tp_size, usp_flag, dp_size, sharding_stage, recompute):
    tp_size_list = [tp_size for _ in range(num_hidden_layers)]
    usp_flag_list = [usp_flag for _ in range(num_hidden_layers)]
    dp_size_list = [dp_size for _ in range(num_hidden_layers)]
    sharding_stage_list = [sharding_stage for _ in range(num_hidden_layers + 1)] # [NOTE] embedding
    recompute_list  = [recompute for _ in range(num_hidden_layers)]
    vtp = tp_size
    vsp_flag = usp_flag

    ave_num_layer = num_hidden_layers // pp_size
    last_num_layer = num_hidden_layers - (pp_size - 1) * ave_num_layer
    pp_stage_idx_list = []
    for i in range(pp_size):
        if i == pp_size - 1:
            pp_stage_idx_list.extend([i] * last_num_layer)
        else:
            pp_stage_idx_list.extend([i] * ave_num_layer)

    return generate_meshs(num_hidden_layers, gpu_nums, pp_size, pp_stage_idx_list, tp_size_list, usp_flag_list, dp_size_list, sharding_stage_list, recompute_list, vtp, vsp_flag)

def generate_meshs_fine_grained(gpu_nums, config_file_path):
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    
    pp_size = int(config['pp_size'])
    dp_size_list = [int(i) for i in config['dp_size_list'].split(',')]
    sharding_stage_list = [int(i) for i in config['sharding_stage_list'].split(',')]
    tp_size_list = [int(i) for i in config['tp_size_list'].split(',')]
    usp_flag_list = [int(i) for i in config['usp_flag_list'].split(',')]
    pp_stage_idx_list = [int(i) for i in config['pp_stage_idx_list'].split(',')]
    recompute_list = [int(i) for i in config['recompute_list'].split(',')]
    vtp = int(config['vtp'])
    vsp_flag = int(config['vsp_flag'])
    embed_sdp = int(config['embed_sdp'])

    if embed_sdp:
        sharding_stage_list = [3] + sharding_stage_list
    else:
        sharding_stage_list = [2] + sharding_stage_list
    
    num_hidden_layers = len(dp_size_list)

    return generate_meshs(num_hidden_layers, gpu_nums, pp_size, pp_stage_idx_list, tp_size_list, usp_flag_list, dp_size_list, sharding_stage_list, recompute_list, vtp, vsp_flag)


def get_redistributed_flag(mesh_list):
    flag = [0] * len(mesh_list)
    for i, mesh in enumerate(mesh_list):
        if i != len(mesh_list) - 1:
            next_mesh_dp_deg = mesh_list[i + 1].shape[0]
            now_hidden_states_dp_deg = mesh.shape[0]
            if next_mesh_dp_deg != now_hidden_states_dp_deg:
                shape = get_dp_tp_ranks(mesh.process_ids, dp_size=next_mesh_dp_deg, tp_size=mesh_list[i + 1].shape[1])
                to_mesh = ProcessMesh(shape, dim_names=mesh.dim_names)
                flag[i] = to_mesh
    return flag
