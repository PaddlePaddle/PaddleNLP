#!/usr/bin/env bash
# -*- coding:utf-8 -*-
CUDA_VISIBLE_DEVICES=0

# data_folder="data/text2spotasoc/event/duee/"
data_folder="data/text2spotasoc/absa/conv_asa"
ordered_prompt=False
seed=42
batch_size=16
lr=5e-4
gradient_accumulation_steps=1
num_train_epochs=50
max_prefix_length=0

model_name="pd_models/uie-char-small"
batch_size_training=$((batch_size / gradient_accumulation_steps))
metric_for_best_model=string-rel-strict-F1
map_config="config/offset_map/longer_first_offset_zh.yaml"

model_name_log=$(echo ${model_name} | sed -s "s/\//_/g")
data_name_log=$(echo ${data_folder} | sed -s "s/\//_/g")

output_dir=output/${model_name_log}_${data_name_log}_b${batch_size}_lr${lr}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3 run_seq2seq_paddle.py \
    --do_train --do_predict                                                \
    --metric_for_best_model=${metric_for_best_model}                       \
    --model_name_or_path=${model_name}                                     \
    --max_source_length=${max_source_length:-"256"}                        \
    --max_prefix_length=${max_prefix_length:-"-1"}                         \
    --max_target_length=${max_target_length:-"192"}                        \
    --num_train_epochs=${num_train_epochs:-"50"}                           \
    --train_file=${data_folder}/train.json                                 \
    --validation_file=${data_folder}/val.json                              \
    --test_file=${data_folder}/test.json                                   \
    --record_schema=${data_folder}/record.schema                           \
    --record_schema_dir=${data_folder}                                     \
    --per_device_train_batch_size=${batch_size_training}                   \
    --per_device_eval_batch_size=$((batch_size_training * 8))              \
    --output_dir=${output_dir}                                             \
    --logging_dir=${output_dir}_log                                        \
    --learning_rate=${lr:-"3e-4"}                                          \
    --lr_scheduler_type=${lr_scheduler:-"linear"}                          \
    --decoding_format ${decoding_format:-"spotasoc"}                       \
    --warmup_ratio ${warmup_ratio:-"0.06"}                                 \
    --dataloader_num_workers=0                                             \
    --meta_negative=${negative:-"-1"}                                      \
    --meta_positive_rate=${positive:-"1"}                                  \
    --spot_noise=${spot_noise:-"0.1"}                                      \
    --asoc_noise=${asoc_noise:-"0.1"}                                      \
    --seed=${seed:-"42"}                                                   \
    --overwrite_output_dir                                                 \
    --gradient_accumulation_steps ${gradient_accumulation_steps}           \
    --config ${map_config}
