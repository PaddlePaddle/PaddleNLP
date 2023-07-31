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


PYTHONPATH=../../:$PYTHONPATH
task_name="ernie_hybrid_sft"

# 多卡微调
python -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "output_sft/$task_name""_log" \
    finetune_generation.py \
    --output_dir "output_sft/$task_name" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --model_name_or_path <PATH_TO_CKPT> \
    --task_name squad \
    --num_train_epochs 2 \
    --learning_rate 3e-5 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --src_length 1024 \
    --tgt_length 1024 \
    --bf16 \
    --fp16_opt_level O2 \
    --do_train \
    --do_eval \
    --disable_tqdm True \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --eval_with_do_generation False \
    --recompute \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --sharding "stage2" \
    --sharding_parallel_degree 8


# 单卡 LoRA 微调
# python -m paddle.distributed.launch \
#      --gpus "0" \
#      --log_dir "output_sft/$task_name""_log" \
#      finetune_generation.py \
#     --output_dir ./checkpoints/ \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 2 \
#     --per_device_eval_batch_size 8 \
#     --model_name_or_path ./output/ernie35_hybid/checkpoint-2400 \
#     --task_name squad \
#     --num_train_epochs 2 \
#     --learning_rate 3e-4 \
#     --warmup_steps 30 \
#     --logging_steps 1 \
#     --evaluation_strategy epoch \
#     --save_strategy epoch \
#     --src_length 1024 \
#     --tgt_length 1024 \
#     --bf16 \
#     --fp16_opt_level O2 \
#     --do_train \
#     --do_eval \
#     --disable_tqdm True \
#     --load_best_model_at_end True \
#     --metric_for_best_model accuracy \
#     --eval_with_do_generation False \
#     --recompute \
#     --save_total_limit 1 \
#     --overwrite_output_dir \
#     --lora True \
#     --lora_rank 8
