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
python -m paddle.distributed.launch --gpus "2,5,6,7" --log_dir ./unimo/finetune/log run_gen.py \
    --dataset_name=dureader_qg \
    --model_name_or_path=/root/project/paddle/drawer/PaddleNLP/applications/question_generation/unimo/pretrain/drawer/train.json-20-best-5e-5/checkpoints/model_20000 \
    --save_dir=./unimo/finetune/checkpoints \
    --output_path ./unimo/finetune/predict.txt \
    --logging_steps=100 \
    --save_steps=200 \
    --epochs=20 \
    --batch_size=16 \
    --learning_rate=1e-5 \
    --warmup_propotion=0.02 \
    --weight_decay=0.01 \
    --max_seq_len=512 \
    --max_target_len=30 \
    --do_train \
    --do_predict \
    --max_dec_len=20 \
    --min_dec_len=3 \
    --num_return_sequences=1 \
    --adversarial_training=None \
    --template=1 \
    --device=gpu
