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
# python -m paddle.distributed.launch --gpus "1,2,3,4" --log_dir ./unimo/finetune/merge9_train_prompt_epoch30/log run_gen.py \
#     --dataset_name=dureader_qg \
#     --train_file=/root/project/data/qa-dataset/qa-clean-qg-merge/merge9_train_prompt.json \
#     --predict_file=/root/project/data/dureader_qg/raw/DuReaderQG/dev_prompt.json \
#     --model_name_or_path=/root/project/paddle/PaddleNLP/applications/question_generation/unimo/unimo-text-1.0-modified_tokenizer \
#     --save_dir=./unimo/finetune/merge9_train_prompt_epoch30/checkpoints \
#     --output_path=./unimo/finetune/merge9_train_prompt_epoch30/predict.txt \
#     --logging_steps=100 \
#     --save_steps=3000 \
#     --epochs=30 \
#     --batch_size=16 \
#     --learning_rate=5e-5 \
#     --warmup_propotion=0.02 \
#     --weight_decay=0.01 \
#     --max_seq_len=512 \
#     --max_target_len=50 \
#     --do_train \
#     --do_predict \
#     --max_dec_len=50 \
#     --min_dec_len=3 \
#     --num_return_sequences=1 \
#     --adversarial_training=None \
#     --template=4 \
#     --device=gpu



unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "1,2" --log_dir ./unimo/finetune/merge9_train_prompt_epoch30_finetune_5e-6/log run_gen.py \
    --dataset_name=dureader_qg \
    --train_file=/root/project/data/dureader_qg/raw/DuReaderQG/train_prompt.json \
    --predict_file=/root/project/data/dureader_qg/raw/DuReaderQG/dev_prompt.json \
    --model_name_or_path=/root/project/paddle/PaddleNLP/applications/question_generation/unimo/finetune/merge9_train_prompt_epoch30/checkpoints/model_best \
    --save_dir=./unimo/finetune/merge9_train_prompt_epoch30_finetune_5e-6/checkpoints \
    --output_path=./unimo/finetune/merge9_train_prompt_epoch30_finetune_5e-6/predict.txt \
    --logging_steps=100 \
    --save_steps=400 \
    --epochs=30 \
    --batch_size=16 \
    --learning_rate=5e-6 \
    --warmup_propotion=0.02 \
    --weight_decay=0.01 \
    --max_seq_len=512 \
    --max_target_len=50 \
    --do_train \
    --do_predict \
    --max_dec_len=50 \
    --min_dec_len=3 \
    --num_return_sequences=1 \
    --adversarial_training=None \
    --template=4 \
    --device=gpu



######################################################################################################################################################################################################################################
# unset CUDA_VISIBLE_DEVICES
# python -m paddle.distributed.launch --gpus "1" --log_dir ./unimo/finetune/dureader_full/log run_gen.py \
#     --dataset_name=dureader_qg \
#     --train_file=/root/project/data/qa-dataset/qa-clean-qg/dureader_data.json \
#     --model_name_or_path='unimo-text-1.0' \
#     --save_dir=./unimo/finetune/dureader_full/checkpoints \
#     --output_path=./unimo/finetune/dureader_full/predict.txt \
#     --logging_steps=100 \
#     --save_steps=400 \
#     --epochs=10 \
#     --batch_size=16 \
#     --learning_rate=5e-5 \
#     --warmup_propotion=0.02 \
#     --weight_decay=0.01 \
#     --max_seq_len=512 \
#     --max_target_len=50 \
#     --do_train \
#     --do_predict \
#     --max_dec_len=40 \
#     --min_dec_len=3 \
#     --num_return_sequences=1 \
#     --adversarial_training=None \
#     --template=1 \
#     --device=gpu



# unset CUDA_VISIBLE_DEVICES
# python -m paddle.distributed.launch --gpus "1" --log_dir ./unimo/finetune/dureader_full_finetune/log run_gen.py \
#     --dataset_name=dureader_qg \
#     --model_name_or_path=/root/project/paddle/PaddleNLP/applications/question_generation/unimo/finetune/dureader_full/checkpoints/model_best \
#     --save_dir=./unimo/finetune/dureader_full_finetune/checkpoints \
#     --output_path=./unimo/finetune/dureader_full_finetune/predict.txt \
#     --logging_steps=100 \
#     --save_steps=400 \
#     --epochs=10 \
#     --batch_size=16 \
#     --learning_rate=5e-5 \
#     --warmup_propotion=0.02 \
#     --weight_decay=0.01 \
#     --max_seq_len=512 \
#     --max_target_len=50 \
#     --do_train \
#     --do_predict \
#     --max_dec_len=40 \
#     --min_dec_len=3 \
#     --num_return_sequences=1 \
#     --adversarial_training=None \
#     --template=1 \
#     --device=gpu
