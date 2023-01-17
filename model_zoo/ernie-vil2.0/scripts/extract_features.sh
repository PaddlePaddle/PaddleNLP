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

# export NCCL_IB_DISABLE=0
export PYTHONPATH=/root/paddlejob/workspace/env_run/wugaosheng/PaddleNLP:$PYTHONPATH
DATAPATH=./data

split=valid # 指定计算valid或test集特征

python -u extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="${DATAPATH}/datasets/Flickr30k-CN/lmdb/${split}/imgs" \
    --text-data="${DATAPATH}/datasets/Flickr30k-CN/${split}_texts.jsonl" \
    --resume output_pd/checkpoint-600 \
    --img-batch-size=32 \
    --text-batch-size=32 \
    --context-length=52 