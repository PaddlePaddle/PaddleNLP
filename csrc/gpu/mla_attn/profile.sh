# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

export CUDA_VISIBLE_DEVICES=7
export bsz=${1}
export kv_seq_len=${2}
export num_heads=${3}
export chunk_size=${4}

ncu=`which ncu`

${ncu} --section regex:'^(?!Nvlink)' -o test_${bsz}_${kv_seq_len}_${num_heads}_${chunk_size}_split -f --import-source on --cache-control=all --clock-control=base -k regex:MLAWithKVCacheKernel --print-source=cuda,sass --page source ./test ${bsz} ${kv_seq_len} ${num_heads} ${chunk_size}