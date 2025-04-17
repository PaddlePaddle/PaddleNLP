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

source /home/pangengzheng/develop/py39/bin/activate

num_attention_heads=32
seq_len=32768
nnodes=1
if [[ $# > 0 ]]; then
  seq_len=$1;
fi


echo "seq_len:${seq_len}"

log_dir=log_hybrid
rm -rf $log_dir

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

# uncommend these for performance
# export FLAGS_embedding_deterministic=1
# export FLAGS_cudnn_deterministic=1
# export FLAGS_flash_attn_version=v1
# export USE_FAST_LN=0

export PYTHONPATH=../../:$PYTHONPATH

rank=${PADDLE_TRAINER_ID-0}
if [[ $nnodes -gt 1 ]]; then
  master=`cat /root/paddlejob/workspace/hostfile | head -n 1 | awk '{print $1}'`
else
  master=127.0.0.1
fi
port=36677

log_dir=seq_${seq_len}_log
echo "log_dir:${log_dir}"
rm -rf $log_dir

python -m paddle.distributed.launch \
    --master $master:$port \
    --nnodes $nnodes \
    --rank $rank \
    --log_dir "./$log_dir" --devices "0,1,2,3,4,5,6,7" \
    ./tools/train.py \
    -c ./ppfleetx/configs/nlp/gpt/pretrain_gpt_1.3B_sep8.yaml \
    -o Engine.save_load.save_steps=10000 \
    -o Engine.max_steps=10 \
    -o Model.use_recompute=False \
    -o Model.num_attention_heads=${num_attention_heads} \
    -o Model.max_position_embeddings=${seq_len} \
    -o Data.Train.dataset.max_seq_len=${seq_len} \
    -o Data.Eval.dataset.max_seq_len=${seq_len} \

