set -x

ProfileDataParserArgs="
    --time_profile_mode batch \
    --memory_profile_mode static \
    --num_layertype 1 \
    --hidden_size_list 4096 \
    --layernum_list 16 \
    --seqlen_list 1024 \
    --profile_gpu_num 8 \
    --time_profile_data_path ./configs/computation_profiling_bf16_llama_rank[0].json \
    --memory_profile_data_path ./configs/memory_profiling_bf16_llama.json \
    --overlap_coe_path ./configs/overlap_coefficient.json \
    --allreduce_coe_path ./configs/allreduce_bandwidth_1nodes_8gpus_per_node.json \
    --p2p_coe_path ./configs/p2p_bandwidth_1nodes_8gpus_per_node.json \
    --sp_time_path ./configs/sp_time_1nodes_8gpus_per_node.json \
"

SearchEngineArgs="
    --search_granularity fine-grained \
    --world_size 8 \
    --min_bsz 128 \
    --max_bsz 128 \
    --bsz_step 1 \
    --max_tp_size 8 \
    --max_pp_size 8 \
    --mixed_precision_type bf16 \
    --memory_upper_limit 40 \
    --sp_space tp+sp \
    --layernum 16 \
    --disable_sdp 0 \
    --disable_vtp 0 \
    --parallel_search 0 \
    --log_dir ./search-engine-logs \
"

python ./search_dist.py ${ProfileDataParserArgs} ${SearchEngineArgs}