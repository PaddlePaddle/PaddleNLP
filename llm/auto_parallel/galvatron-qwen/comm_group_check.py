from paddlenlp.experimental.galvatron.runtime.comm_group import generate_meshs, generate_meshs_coarse_grained

def test():
    gpu_nums = 8
    num_hidden_layers = 16
    pp_size = 2
    
    pp_stage_idx_list = [0 for _ in range(num_hidden_layers // 2)]  + [1 for _ in range(num_hidden_layers // 2)]
    dp_size_list = [2 for _ in range(num_hidden_layers)]
    tp_size_list = [2 for _ in range(num_hidden_layers)]
    usp_flag_list = [0 for _ in range(num_hidden_layers)]
    sharding_stage_list = [0 for _ in range(num_hidden_layers)]
    recompute_stage_list = [0 for _ in range(num_hidden_layers)]
    vtp = 2
    vsp_flag = 0

    # generate_meshs(num_hidden_layers, gpu_nums, pp_size, pp_stage_idx_list, tp_size_list, usp_flag_list, dp_size_list, sharding_stage_list, recompute_stage_list, vtp, vsp_flag)

    generate_meshs_coarse_grained(num_hidden_layers, gpu_nums, pp_size, tp_size_list[0], usp_flag_list[0], dp_size_list[0], sharding_stage_list[0], recompute_stage_list[0])

if __name__ == '__main__':
    test()