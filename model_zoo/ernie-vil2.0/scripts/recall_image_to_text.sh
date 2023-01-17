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
dataset_name=Flickr30k-CN

split=valid # 指定计算valid或test集特征
python -u utils/make_topk_predictions_tr.py \
    --image-feats="${DATAPATH}/datasets/${dataset_name}/${split}_imgs.img_feat.jsonl" \
    --text-feats="${DATAPATH}/datasets/${dataset_name}/${split}_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${DATAPATH}/datasets/${dataset_name}/${split}_tr_predictions.jsonl"

python utils/transform_ir_annotation_to_tr.py \
    --input ${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl

split=valid # 指定计算valid或test集特征
python utils/evaluation_tr.py \
    ${DATAPATH}/datasets/${dataset_name}/${split}_texts.tr.jsonl \
    ${DATAPATH}/datasets/${dataset_name}/${split}_tr_predictions.jsonl \
    output.json
cat output.json