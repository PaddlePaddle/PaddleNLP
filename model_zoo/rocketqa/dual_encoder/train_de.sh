# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

unset CUDA_VISIBLE_DEVICES
TRAIN_SET="../dureader-retrieval-baseline-dataset/train/dual.train.tsv"
python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
                    train_de.py \
                   --train_set_file ${TRAIN_SET} \
                   --save_dir ./checkpoint \
                   --batch_size 128 \
                   --save_steps 8685 \
                   --query_max_seq_length 32 \
                   --title_max_seq_length 384 \
                   --learning_rate 3e-5 \
                   --epochs 10 \
                   --weight_decay 0.0 \
                   --warmup_proportion 0.1 \
                   --use_cross_batch \
                   --seed 1 \
                   --use_amp \
                   --use_recompute