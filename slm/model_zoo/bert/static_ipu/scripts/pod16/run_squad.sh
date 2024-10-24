#!/usr/bin/env bash

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

python3 run_squad.py \
        --output_dir squad_model \
        --task "SQUAD" \
        --is_training True \
        --seq_len 384 \
        --hidden_size 768 \
        --vocab_size 30400 \
        --max_predictions_per_seq 56 \
        --max_position_embeddings 512 \
        --learning_rate 5.6e-05 \
        --weight_decay 0 \
        --epochs 4 \
        --warmup_steps 52 \
        --logging_steps 10 \
        --seed 42 \
        --beta1 0.9 \
        --beta2 0.999 \
        --num_hidden_layers 12 \
        --micro_batch_size 2 \
        --ipu_enable_fp16 True \
        --accl1_type "FLOAT" \
        --accl2_type "FLOAT" \
        --weight_decay_mode "decay" \
        --scale_loss 256 \
        --optimizer_state_offchip False \
        --batches_per_step 4 \
        --num_replica 4 \
        --num_ipus 2 \
        --enable_grad_acc True \
        --grad_acc_factor 16 \
        --available_mem_proportion 0.40 \
        --ignore_index 0 \
        --hidden_dropout_prob 0.1 \
        --attention_probs_dropout_prob 0.1 \
        --shuffle True \
        --wandb False \
        --enable_engine_caching False \
        --enable_load_params True \
        --load_params_path "pretrain_384_model/final_step_2137.pdparams"
