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

# GPU启动，参数`--gpus`指定训练所用的GPU卡号，可以是单卡，也可以多卡
# 例如使用1号和2号卡，则：`--gpu 1,2`
unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus "1,2" --log_dir log/filtration finetune/answer_extraction_and_roundtrip_filtration/finetune.py \
    --train_path=data/finetune/filtration/train.json \
    --dev_path=data/finetune/filtration/dev.json \
    --save_dir=log/filtration/checkpoints \
    --learning_rate=1e-5 \
    --batch_size=16 \
    --max_seq_len=512 \
    --num_epochs=30 \
    --model=uie-base \
    --seed=1000 \
    --logging_steps=100 \
    --valid_steps=5000 \
    --device=gpu

# unset CUDA_VISIBLE_DEVICES
# python -u -m paddle.distributed.launch --gpus "1,2" --log_dir log/question_generation finetune/question_generation/train.py \
#     --train_file=data/finetune/question_generation/train.json \
#     --predict_file=data/finetune/question_generation/dev.json \
#     --save_dir=log/question_generation/checkpoints \
#     --output_path=log/question_generation/predict.txt \
#     --dataset_name=dureader_qg \
#     --model_name_or_path="unimo-text-1.0" \
#     --logging_steps=100 \
#     --save_steps=500 \
#     --epochs=20 \
#     --batch_size=16 \
#     --learning_rate=1e-5 \
#     --warmup_propotion=0.02 \
#     --weight_decay=0.01 \
#     --max_seq_len=512 \
#     --max_target_len=30 \
#     --do_train \
#     --do_predict \
#     --max_dec_len=20 \
#     --min_dec_len=3 \
#     --num_return_sequences=1 \
#     --template=1 \
#     --device=gpu

# # GPU启动，参数`--gpus`指定训练所用的GPU卡号，可以是单卡，也可以多卡
# # 例如使用1号和2号卡，则：`--gpu 1,2`
# unset CUDA_VISIBLE_DEVICES
# python -u -m paddle.distributed.launch --gpus "1,2" --log_dir log/answer_extraction finetune/answer_extraction_and_roundtrip_filtration/finetune.py \
#     --train_path=data/finetune/answer_extraction/train.json \
#     --dev_path=data/finetune/answer_extraction/dev.json \
#     --save_dir=log/answer_extraction/checkpoints \
#     --learning_rate=1e-5 \
#     --batch_size=16 \
#     --max_seq_len=512 \
#     --num_epochs=30 \
#     --model=uie-base \
#     --seed=1000 \
#     --logging_steps=100 \
#     --valid_steps=100 \
#     --device=gpu

# python -u run_data_preprocess.py \
#   --source_file_path data/dev.json \
#   --target_dir data/finetune \
#   --do_answer_prompt

# python -u run_data_preprocess.py \
#     --source_file_path data/train.json \
#     --target_dir data/finetune \
#     --do_answer_prompt

# export CUDA_VISIBLE_DEVICES=0
# python -u run_qa_pairs_generation.py \
#     --source_file_path=data/source_file.txt \
#     --target_file_path=data/target_file.json \
#     --answer_generation_model_path=uie-base-answer-extractor-v1 \
#     --question_generation_model_path=unimo-text-1.0-question-generation \
#     --filtration_model_path=uie-base-qa-filter-v1 \
#     --batch_size=8 \
#     --a_max_answer_candidates=10 \
#     --a_prompt='答案' \
#     --a_position_prob=0.01  \
#     --q_num_return_sequences=3 \
#     --q_max_question_length=50 \
#     --q_decode_strategy=sampling \
#     --q_top_k=5 \
#     --q_top_p=1 \
#     --do_filtration \
#     --f_filtration_position_prob=0.01 \
#     --do_debug