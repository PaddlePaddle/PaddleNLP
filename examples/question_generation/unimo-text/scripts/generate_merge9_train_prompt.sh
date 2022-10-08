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


export CUDA_VISIBLE_DEVICES=4

python -u run_gen.py \
    --dataset_name=dureader_qg \
    --model_name_or_path=/root/project/paddle/PaddleNLP/applications/question_generation/unimo/finetune/merge9_train_prompt_epoch30/checkpoints/model_best \
    --output_path ./unimo/generate/merge9_train_prompt_epoch30_finetune_zeroshot_dureader_qg_predict.txt \
    --logging_steps=100 \
    --batch_size=16 \
    --max_seq_len=512 \
    --max_target_len=50 \
    --do_predict \
    --max_dec_len=50 \
    --min_dec_len=3 \
    --template=4 \
    --device=gpu \
    --predict_file=/root/project/data/dureader_qg/raw/DuReaderQG/dev_prompt.json \

python -u run_gen.py \
    --dataset_name=dureader_qg \
    --model_name_or_path=/root/project/paddle/PaddleNLP/applications/question_generation/unimo/finetune/merge9_train_prompt_epoch30/checkpoints/model_best \
    --output_path ./unimo/generate/merge9_train_prompt_epoch30_finetune_zeroshot_cail_data_predict.txt \
    --logging_steps=100 \
    --batch_size=16 \
    --max_seq_len=512 \
    --max_target_len=50 \
    --do_predict \
    --max_dec_len=50 \
    --min_dec_len=3 \
    --template=4 \
    --device=gpu \
    --predict_file=/root/project/data/qa-dataset/prompt-qa-cleran-qg-for-test/cail_data.json \

python -u run_gen.py \
    --dataset_name=dureader_qg \
    --model_name_or_path=/root/project/paddle/PaddleNLP/applications/question_generation/unimo/finetune/merge9_train_prompt_epoch30/checkpoints/model_best \
    --output_path ./unimo/generate/merge9_train_prompt_epoch30_finetune_zeroshot_cmrc_data_predict.txt \
    --logging_steps=100 \
    --batch_size=16 \
    --max_seq_len=512 \
    --max_target_len=50 \
    --do_predict \
    --max_dec_len=50 \
    --min_dec_len=3 \
    --template=4 \
    --device=gpu \
    --predict_file=/root/project/data/qa-dataset/prompt-qa-cleran-qg-for-test/cmrc_data.json \

python -u run_gen.py \
    --dataset_name=dureader_qg \
    --model_name_or_path=/root/project/paddle/PaddleNLP/applications/question_generation/unimo/finetune/merge9_train_prompt_epoch30/checkpoints/model_best \
    --output_path ./unimo/generate/merge9_train_prompt_epoch30_finetune_zeroshot_drcd_data_predict.txt \
    --logging_steps=100 \
    --batch_size=16 \
    --max_seq_len=512 \
    --max_target_len=50 \
    --do_predict \
    --max_dec_len=50 \
    --min_dec_len=3 \
    --template=4 \
    --device=gpu \
    --predict_file=/root/project/data/qa-dataset/prompt-qa-cleran-qg-for-test/drcd_data.json \

python -u run_gen.py \
    --dataset_name=dureader_qg \
    --model_name_or_path=/root/project/paddle/PaddleNLP/applications/question_generation/unimo/finetune/merge9_train_prompt_epoch30/checkpoints/model_best \
    --output_path ./unimo/generate/merge9_train_prompt_epoch30_finetune_zeroshot_dureader_data_predict.txt \
    --logging_steps=100 \
    --batch_size=16 \
    --max_seq_len=512 \
    --max_target_len=50 \
    --do_predict \
    --max_dec_len=50 \
    --min_dec_len=3 \
    --template=4 \
    --device=gpu \
    --predict_file=/root/project/data/qa-dataset/prompt-qa-cleran-qg-for-test/dureader_data.json \

python -u run_gen.py \
    --dataset_name=dureader_qg \
    --model_name_or_path=/root/project/paddle/PaddleNLP/applications/question_generation/unimo/finetune/merge9_train_prompt_epoch30/checkpoints/model_best \
    --output_path ./unimo/generate/merge9_train_prompt_epoch30_finetune_zeroshot_medicine_data_predict.txt \
    --logging_steps=100 \
    --batch_size=16 \
    --max_seq_len=512 \
    --max_target_len=50 \
    --do_predict \
    --max_dec_len=50 \
    --min_dec_len=3 \
    --template=4 \
    --device=gpu \
    --predict_file=/root/project/data/qa-dataset/prompt-qa-cleran-qg-for-test/medicine_data.json \

python -u run_gen.py \
    --dataset_name=dureader_qg \
    --model_name_or_path=/root/project/paddle/PaddleNLP/applications/question_generation/unimo/finetune/merge9_train_prompt_epoch30/checkpoints/model_best \
    --output_path ./unimo/generate/merge9_train_prompt_epoch30_finetune_zeroshot_military_data_predict.txt \
    --logging_steps=100 \
    --batch_size=16 \
    --max_seq_len=512 \
    --max_target_len=50 \
    --do_predict \
    --max_dec_len=50 \
    --min_dec_len=3 \
    --template=4 \
    --device=gpu \
    --predict_file=/root/project/data/qa-dataset/prompt-qa-cleran-qg-for-test/military_data.json \

python -u run_gen.py \
    --dataset_name=dureader_qg \
    --model_name_or_path=/root/project/paddle/PaddleNLP/applications/question_generation/unimo/finetune/merge9_train_prompt_epoch30/checkpoints/model_best \
    --output_path ./unimo/generate/merge9_train_prompt_epoch30_finetune_zeroshot_squad_data_predict.txt \
    --logging_steps=100 \
    --batch_size=16 \
    --max_seq_len=512 \
    --max_target_len=50 \
    --do_predict \
    --max_dec_len=50 \
    --min_dec_len=3 \
    --template=4 \
    --device=gpu \
    --predict_file=/root/project/data/qa-dataset/prompt-qa-cleran-qg-for-test/squad_data.json \

python -u run_gen.py \
    --dataset_name=dureader_qg \
    --model_name_or_path=/root/project/paddle/PaddleNLP/applications/question_generation/unimo/finetune/merge9_train_prompt_epoch30/checkpoints/model_best \
    --output_path ./unimo/generate/merge9_train_prompt_epoch30_finetune_zeroshot_webqa_data_predict.txt \
    --logging_steps=100 \
    --batch_size=16 \
    --max_seq_len=512 \
    --max_target_len=50 \
    --do_predict \
    --max_dec_len=50 \
    --min_dec_len=3 \
    --template=4 \
    --device=gpu \
    --predict_file=/root/project/data/qa-dataset/prompt-qa-cleran-qg-for-test/webqa_data.json \

python -u run_gen.py \
    --dataset_name=dureader_qg \
    --model_name_or_path=/root/project/paddle/PaddleNLP/applications/question_generation/unimo/finetune/merge9_train_prompt_epoch30/checkpoints/model_best \
    --output_path ./unimo/generate/merge9_train_prompt_epoch30_finetune_zeroshot_yiqing_data_predict.txt \
    --logging_steps=100 \
    --batch_size=16 \
    --max_seq_len=512 \
    --max_target_len=50 \
    --do_predict \
    --max_dec_len=50 \
    --min_dec_len=3 \
    --template=4 \
    --device=gpu \
    --predict_file=/root/project/data/qa-dataset/prompt-qa-cleran-qg-for-test/yiqing_data.json \

