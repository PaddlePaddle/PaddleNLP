export CUDA_VISIBLE_DEVICES=4,5,6,7

output_dir="vllm_bf16_offline_flashattn"

nohup python torch_infer.py \
    --actor_model_name_or_path your_model_name \
    --max_src_len 2048 \
    --min_dec_len 32 \
    --max_dec_len 30720 \
    --top_p 1.0 \
    --temperature 1.0 \
    --rollout_input_batch_size 4 \
    --rollout_n 8 \
    --tensor_parallel_degree 4 \
    --limit_rows 640 \
    --input_file file.parquet \
    --output_dir ${output_dir} \
    --gpu_memory_utilization 0.8 > ./torchinferflashattn.log 2>&1 &