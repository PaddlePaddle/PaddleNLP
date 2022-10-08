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
python -m paddle.distributed.launch --gpus "1,5,6,7" --log_dir ./unimo/pretrain/log pretrain.py \
    --train_file=/root/project/data/dureader/acsg_pretrain/train.json \
    --predict_file=/root/project/data/dureader/acsg_pretrain/dev_mini.json \
    --model_name_or_path='unimo-text-1.0' \
    --save_dir=./unimo/pretrain/checkpoints \
    --output_path ./unimo/pretrain/predict.txt \
    --logging_steps=1000 \
    --save_steps=10000 \
    --epochs=30 \
    --batch_size=16 \
    --learning_rate=5e-6 \
    --warmup_propotion=0.02 \
    --weight_decay=0.01 \
    --max_seq_len=512 \
    --max_target_len=80 \
    --do_pretrain \
    --do_predict \
    --max_dec_len=80 \
    --min_dec_len=3 \
    --num_return_sequences=1 \
    --adversarial_training=None \
    --device=gpu