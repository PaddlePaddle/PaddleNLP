# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# gpu
python -u -m paddle.distributed.launch --gpus "0" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "checkpoints/model_12000/model_state.pdparams" \
        --model_name_or_path rocketqa-zh-base-query-encoder \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "recall/dev.csv" \
        --corpus_file "recall/corpus.csv" 

# cpu
# python  recall.py \
#         --device cpu \
#         --recall_result_dir "recall_result_dir" \
#         --recall_result_file "recall_result.txt" \
#         --params_path "checkpoints/model_20000/model_state.pdparams" \
#         --hnsw_m 100 \
#         --hnsw_ef 100 \
#         --batch_size 64 \
#         --output_emb_size 256\
#         --max_seq_length 60 \
#         --recall_num 50 \
#         --similar_text_pair "recall/dev.csv" \
#         --corpus_file "recall/corpus.csv" 