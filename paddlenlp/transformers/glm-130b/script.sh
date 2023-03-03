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

python run_train.py 

    --micro_batch_size 1 \
    --global_batch_size 4224 \
    --rampup_batch_size 192 24 $batch_warmup_samples \

    --tensor_model_parallel_size 4 \
    --pipeline_model_parallel_size 8 \
    --pp_partition_method "type:transformer|embedding" \

    --save_steps 100 \
    --logging_steps 1 \
    --eval_steps 1000 \

    
    --eval_iters 3 \
    --num_train_tokens 450000000000 \
    --num_train_samples 450000000000 / $seq_length \

    --lr_decay_ratio 0.9
    --lr_warmup_ratio 0.005
    --batch_warmup_ratio 0.025

    --zero_stage 1 \

    --optim adamw \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 4e-5 \
    --min-lr 4e-6 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 0.1 \

    --gpt_prob 0.7 \
    --single_span_prob 0.02 \
    --short_seq_prob 0.02 \
    --mask_prob 0.15 \
    --average_block_length 3 \
    --min-gmask-ratio 0.2 \
    --aggregated-samples-per-sequence 4 \
    --deepnorm \
    --position-embedding-type rotary \
    --glu-activation geglu \
    --no-bias-gelu-fusion \
    --partition-activations \

    --multitask-ratio 0.05 \
    --abort-on-unmet-fused-kernel-constraints \
    --split 949,50,1 \
    --init-method-std 0.0052 \
    --recompute \
    --shrink-logit-embedding-gradient \
    --shrink-embedding-gradient-alpha 0.1 \
    --fp16 \
    
