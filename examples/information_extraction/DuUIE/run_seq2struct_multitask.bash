#!/usr/bin/env bash
# -*- coding:utf-8 -*-
export PYTHONPATH="${PYTHONPATH}:./"
CUDA_VISIBLE_DEVICES=0


data_folder="data/duuie_pre/"
ordered_prompt=False
seed=42
batch_size=32
lr=5e-4
gradient_accumulation_steps=1
num_train_epochs=10

model_name="pd_models/uie-char-small"
batch_size_training=$((batch_size / gradient_accumulation_steps))
metric_for_best_model=all-task-ave

output_dir=output/duuie_multi_task_b${batch_size}_lr${lr}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3              \
    run_seq2struct_multitask.py                                   \
    --do_train                                                    \
    --metric_for_best_model=${metric_for_best_model}              \
    --model_name_or_path=${model_name}                            \
    --max_source_length=${max_source_length:-"384"}               \
    --max_prefix_length=${max_prefix_length:-"-1"}                \
    --max_target_length=${max_target_length:-"192"}               \
    --num_train_epochs=${num_train_epochs:-"50"}                  \
    --train_file=${data_folder}/train.json                        \
    --validation_file=${data_folder}/val.json                     \
    --record_schema=${data_folder}/record.schema                  \
    --per_device_train_batch_size=${batch_size_training}          \
    --per_device_eval_batch_size=$((batch_size_training * 8))     \
    --output_dir=${output_dir}                                    \
    --logging_dir=${output_dir}_log                               \
    --learning_rate=${lr:-"3e-4"}                                 \
    --lr_scheduler_type=${lr_scheduler:-"linear"}                 \
    --warmup_ratio ${warmup_ratio:-"0.06"}                        \
    --dataloader_num_workers=0                                    \
    --meta_negative=${negative:-"-1"}                             \
    --meta_positive_rate=${positive:-"1"}                         \
    --spot_noise=${spot_noise:-"0.1"}                             \
    --asoc_noise=${asoc_noise:-"0.1"}                             \
    --seed=${seed:-"42"}                                          \
    --overwrite_output_dir                                        \
    --gradient_accumulation_steps ${gradient_accumulation_steps}  \
    --multi_task                                                  \
    --multi_task_config config/multitask/multi-task-duuie.yaml    \
    --negative_keep 0.1
