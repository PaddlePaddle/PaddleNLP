# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

ps aux | grep run_pretrain.py | grep -v grep | awk '{print $2}' | xargs kill -9

task_name="llama_hybrid"
rm -rf output/$task_name/
rm -rf "output/$task_name""_log"

PYTHONPATH=..:$PYTHONPATH \
python -m paddle.distributed.launch \
    --gpus '0,1,2,3,4,5,6,7' \
    --log_dir "output/$task_name""_log" \
    run_pretrain.py \
    --model_name_or_path  "meta-llama/Llama-2-13b" \
    --tokenizer_name_or_path  "meta-llama/Llama-2-13b" \
    --input_dir  "llama/data" \
    --output_dir  "llama/output" \
    --per_device_train_batch_size  1 \
    --gradient_accumulation_steps  1024 \
    --per_device_eval_batch_size  64 \
    --tensor_parallel_degree  1 \
    --pipeline_parallel_degree  8 \
    --pipeline_parallel_config  "disable_partial_send_recv" \
    --sharding_parallel_degree  -1 \
    --virtual_pp_degree  1 \
    --sharding  "stage1" \
    --sequence_parallel  0 \
    --adam_beta1  0.9 \
    --adam_beta2  0.95 \
    --use_flash_attention  true \
    --use_fused_rms_norm  true \
    --use_fused_rope  true \
    --max_seq_length  4096 \
    --learning_rate  1e-04 \
    --min_learning_rate  1e-05 \
    --warmup_steps  2000 \
    --logging_steps 1 \
    --max_steps  200000 \
    --save_steps  500 \
    --eval_steps  2000 \
    --weight_decay  0.1 \
    --max_grad_norm  1.0 \
    --amp_master_grad 1 \
    --fp16  true \
    --fp16_opt_level  "O2" \
    --dataloader_num_workers 1 \
    --continue_training 0 \
    --do_train  true \
    --do_eval  true \
    --do_predict  true \
    --disable_tqdm  true \
    --recompute  false \
    --distributed_dataloader 0 \
    --recompute_granularity  "full" \
    --save_total_limit 10