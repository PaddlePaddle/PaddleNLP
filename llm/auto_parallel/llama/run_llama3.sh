# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# just for debug

set -x
unset CUDA_VISIBLE_DEVICES

task_name="llama3_dp2pp4sd2"
rm -rf output/$task_name/
rm -rf "output/$task_name""_log"

export SOT_LOG_LEVEL=4
export PYTHONPATH=../../../:$PYTHONPATH

#ulimit -c unlimited
#export GLOG_v=10

# export FLAGS_call_stack_level=3
# export FLAGS_use_cuda_managed_memory=true

# export FLAGS_embedding_deterministic=1        
# export FLAGS_cudnn_deterministic=1
# export NVIDIA_TF32_OVERRIDE=0

python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir  "output/$task_name""_log" \
    ./run_pretrain_auto.py \
    --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --tokenizer_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --input_dir "./data" \
    --output_dir "./output" \
    --split 949,50,1 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --learning_rate 3e-05 \
    --min_learning_rate 3e-06 \
    --max_steps 30 \
    --logging_steps 10 \
    --eval_steps 1000 \
    --save_steps 50000 \
    --continue_training 0 \
    --do_train true \
    --do_eval false \
    --do_predict false \
    --disable_tqdm true \
    --skip_profile_timer true \
    --save_total_limit 2 \
    --device gpu \
    --disable_tqdm true \
    --dataloader_num_workers 1 \
    --distributed_dataloader 0 \
    --enable_auto_parallel 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 1 \
    --recompute false \
    --recompute_use_reentrant true \
    --recompute_granularity full \
    --pp_recompute_interval 0 \
    --bf16 true \
    --fp16_opt_level "O2"  \
    --amp_master_grad true \
    --fuse_attention_ffn false \
    --fuse_attention_qkv false \
    --fused_linear_param_grad_add 1 \
    --fuse_sequence_parallel_allreduce false \
    --use_flash_attention true \
    --use_fused_rope true \
    --use_fused_rms_norm true \
    --max_seq_length 4096 \
    --sep_parallel_degree 1 \
    --sequence_parallel false \
    --pipeline_parallel_degree 2 \
    --sharding_parallel_degree 2 \
    --tensor_parallel_degree 1 \
    --virtual_pp_degree 4 \
    --pipeline_schedule_mode "VPP" \
    --sharding "stage2" \
    --pipeline_parallel_config "enable_send_recv_overlap" \
    --data_parallel_config "enable_allreduce_avg_in_gradinent_scale gradient_sync_after_accumulate" \
    --sharding_parallel_config "enable_overlap" \
    --tensor_parallel_config "enable_mp_async_allreduce" \
    --to_static 1 \
    --amp_custom_black_list "reduce_sum" "c_softmax_with_cross_entropy" \
    --amp_custom_white_list "lookup_table" "lookup_table_v2" \
    --skip_memory_metrics 0 

