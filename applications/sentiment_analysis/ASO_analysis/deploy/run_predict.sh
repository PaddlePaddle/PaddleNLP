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

export CUDA_VISIBLE_DEVICES=0

python  predict.py \
        --base_model_name "skep_ernie_1.0_large_ch" \
        --ext_model_path "../checkpoints/ext_checkpoints/static/infer" \
        --cls_model_path "../checkpoints/cls_checkpoints/static/infer" \
        --ext_label_path "../data/ext_data/label.dict" \
        --cls_label_path "../data/cls_data/label.dict" \
        --test_path "../data/test.txt" \
        --save_path "../data/sentiment_results.json" \
        --batch_size 8 \
        --ext_max_seq_len 512 \
        --cls_max_seq_len 256
