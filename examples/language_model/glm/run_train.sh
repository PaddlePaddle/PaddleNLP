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

device=$1
model="./local-2b"

CUDA_VISIBLE_DEVICES=$device python run_train.py \
--model_name_or_path $model \
--task_name cnn_dm \
--data_path ./cnn_dm \
--num_train_epochs 4 \
--learning_rate 3e-5 \
--warmup_ratio 0.06 \
--weight_decay 0.1 \
--label_smoothing 0.1 \
--save_steps 10000 \
--logging_steps 1 \
--eval_steps 4 \
--output_dir ./checkpoints/glm-$model-cnn_dm \
--src_length 608 \
--tgt_length 160 \
--min_tgt_length 55 \
--length_penalty 0.7 \
--no_repeat_ngram_size 3 \
--num_beams 5 \
--select_topk True \
--per_device_eval_batch_size 1 \
--per_device_train_batch_size 1 \
--max_grad_norm 1.0 \
--lr_scheduler_type linear \
--do_train
--do_eval 

#--resume_from_checkpoint ../../../paddlenlp/transformers/glm/paddle/glm-515m/ 
#--per_device_eval_batch_size 4 \
#--logging_steps 50 \
#--eval_steps 1000 \
--fp16 \
--recompute \
