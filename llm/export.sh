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

export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH


model_dir=${1:-"checkpoints/llama65b_ptq_smooth"}
src_len=${2:-1100}
dec_len=${3:-330}
quant_type=${4:-"a8w8"}

total_len=`expr ${src_len} + ${dec_len}`



python -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
     export_model.py \
    --model_name_or_path ${model_dir} \
    --output_path ./inference_model/${model_dir}_mp8 \
    --dtype float16 \
    --inference_model \
    --block_size 64 \
    --src_length ${total_len} \
    --block_attn \
    --quant_type ${quant_type} \
    --use_cachekv_int8 static