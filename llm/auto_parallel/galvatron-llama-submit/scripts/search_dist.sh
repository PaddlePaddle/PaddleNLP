set -x

ProfileDataParserArgs="
    --time_profile_mode sequence \
    --memory_profile_mode static \
    --num_layertype 1 \
    --hidden_size_list 5120 \
    --layernum_list 72 \
    --seqlen_list 32768 \
    --profile_gpu_num 64 \
    --time_profile_data_path ./configs/computation_profiling_bf16_llama_rank[0].json \
    --memory_profile_data_path ./configs/memory_profiling_bf16_llama.json \
    --overlap_coe_path ./configs/overlap_coefficient.json \
    --allreduce_coe_path ./configs/allreduce_bandwidth_8nodes_8gpus_per_node.json \
    --p2p_coe_path ./configs/p2p_bandwidth_8nodes_8gpus_per_node.json \
    --sp_time_path ./configs/sp_time_8nodes_8gpus_per_node.json \
"

SearchEngineArgs="
    --search_granularity fine-grained \
    --world_size 64 \
    --min_bsz 64 \
    --max_bsz 64 \
    --bsz_step 1 \
    --max_tp_size 8 \
    --max_pp_size 4 \
    --mixed_precision_type bf16 \
    --memory_upper_limit 95 \
    --sp_space tp \
    --layernum 72 \
    --disable_sdp 0 \
    --disable_vtp 0 \
    --parallel_search 0 \
    --log_dir ./search-engine-logs \
"

python ./search_dist.py ${ProfileDataParserArgs} ${SearchEngineArgs}