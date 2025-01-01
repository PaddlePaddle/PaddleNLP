import json
import os
from .strategy_utils import form_strategy
from typing import List
import numpy as np
from scipy.optimize import curve_fit

def str2array(s):
    return list(map(int,s.split(',')))

def array2str(a):
    return ",".join(map(str,a))

def read_json_config(path):
    return json.load(open(path,'r',encoding="utf-8"))

def write_json_config(config, path):
    with open(path,'w') as fp:
        json.dump(config,fp, indent=4)

def config2strategy(config):
    pp_deg = config['pp_deg']
    if 'vtp' in config:
        vtp = config['vtp']
    else:
        vtp = 1
    if 'vsp' in config:
        vsp = config['vsp']
    else:
        vsp = 0
    tp_sizes_enc = str2array(config['tp_sizes_enc'])
    tp_consecutive_flags = str2array(config['tp_consecutive_flags'])
    dp_types_enc = str2array(config['dp_types_enc'])
    if "use_sp" in config:
        use_sp = str2array(config['use_sp'])
    else:
        use_sp = [0 for _ in range(len(tp_sizes_enc))]
    return pp_deg, tp_sizes_enc, tp_consecutive_flags, dp_types_enc, use_sp, vtp, vsp

def strategy2config(strategy_list):
    layer_num = len(strategy_list)
    if layer_num == 0:
        return {}
    pp_deg = strategy_list[0][0]
    tp_sizes_enc = array2str([s[1] for s in strategy_list])
    tp_consecutive_flags = array2str([0 if 'tp' in s[-1] and not s[-1]['tp'] else 1 for s in strategy_list])
    dp_types_enc = array2str([1 if 'fsdp' in s[-1] and s[-1]['fsdp'] else 0 for s in strategy_list])
    sp = array2str([1 if 'sp' in s[-1] and s[-1]['sp'] else 0 for s in strategy_list])
    
    config = {"pp_deg":pp_deg, "tp_sizes_enc":tp_sizes_enc, "tp_consecutive_flags":tp_consecutive_flags, "dp_types_enc":dp_types_enc, "use_sp":sp}
    return config

def read_allreduce_bandwidth_config(config_path, gpu_num):
    env_config = read_json_config(config_path)
    comm_coe_dict, bandwidth_dict = {}, {}
    max_dp = gpu_num
    if max_dp >= 2:
        bandwidth_dict['%d'%max_dp]=env_config['allreduce_size_%d_consec_1'%(max_dp)]
        comm_coe_dict['%d'%max_dp]=1.0/bandwidth_dict['%d'%max_dp]
    max_dp = max_dp // 2
    while max_dp >= 2:
        bandwidth_dict['%d_0'%max_dp]=env_config['allreduce_size_%d_consec_0'%(max_dp)]
        comm_coe_dict['%d_0'%max_dp]=1.0/bandwidth_dict['%d_0'%max_dp]
        bandwidth_dict['%d_1'%max_dp]=env_config['allreduce_size_%d_consec_1'%(max_dp)]
        comm_coe_dict['%d_1'%max_dp]=1.0/bandwidth_dict['%d_1'%max_dp]
        max_dp = max_dp // 2
    bandwidth_dict['1']=np.inf
    comm_coe_dict['1']=0
    return bandwidth_dict, comm_coe_dict

def read_p2p_bandwidth_config(config_path):
    env_config = read_json_config(config_path)
    pp_deg = 2
    p2p_dict,comm_coe_dict={},{}
    for key, val in env_config.items():
        if 'pp_size_' in key:
            p2p_dict[int(key.split('_')[-1])] = val
            comm_coe_dict[int(key.split('_')[-1])] = 1.0/val
    return p2p_dict, comm_coe_dict

def save_profiling_results(path, strategy, bsz, hidden_size, results):
    config = read_json_config(path) if os.path.exists(path) else {}
    key = form_strategy(strategy)
    if key not in config.keys():
        config[key] = {}
    config[key]['hidden%d_bsz%d'%(hidden_size, bsz)] = results
    write_json_config(config, path)
    print('Already written policy profiling config into config file %s!\n'%(path)) 

def layernum2str(layer_num):
    if isinstance(layer_num, List):
        layernum_info = 'layernum[%s]'%(array2str(layer_num))
    else:
        layernum_info = 'layernum%d'%layer_num
    return layernum_info

def save_profiled_memory(path, pp_deg, tp_deg, world_size, layer_num, bsz, rank, model_states, activation, activation_peak, cpt, sequence_parallel = False, vocab_tp = 1, seq = None):
    config = read_json_config(path) if os.path.exists(path) else {}
    key = '%d_%d_%d'%(pp_deg,tp_deg,world_size//pp_deg//tp_deg)
    if cpt:
        key += '_c'
    if vocab_tp == tp_deg and tp_deg != 1:
        key += '_vtp'
    if sequence_parallel:
        key += '_sp'
    if key not in config.keys():
        config[key] = {}
    layernum_info = layernum2str(layer_num)
    config[key]['%s_bsz%d_seq%d_rank%d_ms'%(layernum_info, bsz, seq, rank)] = model_states
    config[key]['%s_bsz%d_seq%d_rank%d_act'%(layernum_info, bsz, seq, rank)] = activation
    config[key]['%s_bsz%d_seq%d_rank%d_act_peak'%(layernum_info, bsz, seq, rank)] = activation_peak
    write_json_config(config, path)
    print('Already written profiled memory into config file %s!\n'%(path)) 
     
def save_profiled_time(path, time, bsz, layer_num, seq):
    config = read_json_config(path) if os.path.exists(path) else {}
    layernum_info = layernum2str(layer_num)
    key = '%s_bsz%d_seq%d'%(layernum_info, bsz, seq)
    config[key] = time
    write_json_config(config, path)
    print('Already written profiled time into config file %s!\n'%(path)) 
    
def dict_join_dirname(dic, dirname):
    for key, val in dic.items():
        dic[key] = os.path.join(dirname, val)
    return dic

def remap_config(config, op):
    remap_config = {}
    for key, val in config.items():
        if key.startswith(op):
            split = key.split("_")
            world_size, size = int(split[-3]), int(split[-2][:-2])
            if world_size in remap_config:
                remap_config[world_size][size * 1024 * 1024] = val
            else:
                remap_config[world_size] = {}
                remap_config[world_size][size * 1024 * 1024] = val
    
    for world_size, time_config in remap_config.items():
        x_data = []
        y_data = []
        for size, time in time_config.items():
            x_data.append(size // 1024 // 1024)
            y_data.append(time)
        assert len(x_data) >= 8, f"Different size in communication profile of {op} should not be lower than 8."
    
        def linear_func(x, m, c):
            return m * x + c
        popt, pcov = curve_fit(linear_func, x_data, y_data)
        
        print(f"Fitted parameters of {op}", popt)
        
        time_config["popt"] = popt
        
    return remap_config