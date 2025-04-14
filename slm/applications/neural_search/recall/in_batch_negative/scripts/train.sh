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
 
root_dir="./checkpoints/inbatch" 

if [ ! -d "$root_dir" ]; then
    mkdir -p "$root_dir"
    echo "Created directory: $root_dir"
else
    echo "Directory already exists: $root_dir"
fi

python -u -m paddle.distributed.launch --gpus "0" \
    train_batch_neg.py \
    --device gpu \
    --save_dir ${root_dir} \
    --batch_size 64 \
    --learning_rate 5E-5 \
    --epochs 3 \
    --output_emb_size 256 \
    --model_name_or_path rocketqa-zh-base-query-encoder \
    --save_steps 10 \
    --max_seq_length 64 \
    --margin 0.2 \
    --train_set_file recall/train.csv \
    --recall_result_dir "recall_result_dir" \
    --recall_result_file "recall_result.txt" \
    --hnsw_m 100 \
    --hnsw_ef 100 \
    --recall_num 50 \
    --similar_text_pair_file "recall/dev.csv" \
    --corpus_file "recall/corpus.csv"