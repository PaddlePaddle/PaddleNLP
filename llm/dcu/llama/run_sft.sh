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

ps aux | grep run_finetune.py | grep -v grep | awk '{print $2}' | xargs kill -9

PYTHONPATH=..:$PYTHONPATH \
python run_finetune.py \
    --model_name_or_path "meta-llama/Llama-2-13b" \
    --dataset_name_or_path "./data" \
    --output_dir "./checkpoints/llama_sft_ckpts" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --per_device_eval_batch_size 8 \
    --eval_accumulation_steps 16 \
    --num_train_epochs 1 \
    --learning_rate 3e-05 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --eval_step 80000 \
    --save_step 80000 \
    --src_length 1024 \
    --max_length 2048 \
    --fp16 true \
    --fp16_opt_level "O2" \
    --do_train true \
    --do_eval true \
    --disable_tqdm true \
    --load_best_model_at_end true \
    --eval_with_do_generation true \
    --metric_for_best_model "accuracy" \
    --recompute false \
    --save_total_limit 1 \
    --tensor_parallel_degree 1 \
    --pipeline_parallel_degree 1 \
    --intokens false \
    --zero_padding false \
    --use_flash_attention false