unset PADDLE_TRAINERS_NUM
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

export PYTHONPATH="your_paddlenlp_path/PaddleNLP":$PYTHONPATH
export PYTHONPATH="your_paddlenlp_path/PaddleNLP/llm":$PYTHONPATH

export FLAGS_set_to_1d=False
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_dataloader_use_file_descriptor=False
export HF_DATASETS_DOWNLOAD_TIMEOUT=1
export FLAGS_gemm_use_half_precision_compute_type=False
export FLAGS_force_cublaslt_no_reduced_precision_reduction=True

export FLAGS_custom_allreduce=0
export FLAGS_mla_use_tensorcore=0
export FLAGS_cascade_attention_max_partition_size=2048

export CUDA_VISIBLE_DEVICES=4,5,6,7
output_dir="pdpd_bf16_offline"

python -u -m paddle.distributed.launch --log_dir ${output_dir}/logs --gpus ${CUDA_VISIBLE_DEVICES} paddle_infer.py \
  --actor_model_name_or_path your_model_name \
  --max_src_len 2048 \
  --min_dec_len 32 \
  --max_dec_len 30720 \
  --top_p 1.0 \
  --temperature 1.0 \
  --rollout_input_batch_size 4 \
  --rollout_n 8 \
  --rollout_max_num_seqs 24 \
  --rollout_quant_type "" \
  --tensor_parallel_degree 4 \
  --limit_rows 640 \
  --input_file file.parquet \
  --output_dir ${output_dir} > ./paddleinfer.log 2>&1 