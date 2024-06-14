
#! /bin/bash

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


export CUDA_VISIBLE_DEVICES=0


# 导出可验证模型
# python ./tools/export.py \
#     -c ./ppfleetx/configs/nlp/gpt/export_qat_gpt_345M_single_card.yaml \
#     -o Model.hidden_dropout_prob=0.0 \
#     -o Model.attention_probs_dropout_prob=0.0 \
#     -o Engine.save_load.ckpt_dir='./GPT_345M_QAT_w_analysis/'

# 导出可生成句子模型
python ./tools/export.py \
    -c ./ppfleetx/configs/nlp/gpt/generation_qat_gpt_345M_single_card.yaml \
    -o Model.hidden_dropout_prob=0.0 \
    -o Model.attention_probs_dropout_prob=0.0 \
    -o Engine.save_load.ckpt_dir='./GPT_345M_QAT_wo_analysis/'
