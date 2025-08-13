set -x

ProfileDataParserArgs="
    --time_profile_mode sequence \
    --memory_profile_mode sequence \
    --num_layertype 1 \
    --hidden_size_list 8192 \
    --layernum_list 4 \
    --seqlen_list 8192 \
    --profile_gpu_num 8 \
    --time_profile_data_path ./configs/computation_profiling_bf16_llama_rank[0].json \
    --memory_profile_data_path ./configs/memory_profiling_bf16_llama.json \
    --overlap_coe_path ./configs/overlap_coefficient.json \
    --allreduce_coe_path ./configs/allreduce_bandwidth_1nodes_8gpus_per_node-nccl.json \
    --p2p_coe_path ./configs/p2p_bandwidth_1nodes_8gpus_per_node-nccl.json \
    --sp_time_path ./configs/sp_time_1nodes_8gpus_per_node-nccl.json \
"

CostModelTrainArgs="
    --strategy pp1_tp2_dp4_stage2_recompute0 \
    --global_batch_size 32 \
    --mixed_precision_type bf16 \
    --accumulation_steps 2 \
"

/apdcephfs_fsgm/share_303760348/anaconda3/envs/lgm-paddle/bin/python ./check_cost_model.py ${ProfileDataParserArgs} ${CostModelTrainArgs}