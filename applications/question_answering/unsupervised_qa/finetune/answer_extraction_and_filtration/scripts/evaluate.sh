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

# The checkpoint paths of the filter model to load
model_paths[0]=

# Data paths used for testing
dev_paths[0]=


limit=0.01
###############################################################################################################################################################################################################################################################################################################################################
i_length=${#dev_paths[*]}
for ((i=0; i<i_length; i++))
do
    j_length=${#model_paths[*]}
    for ((j=0; j<j_length; j++))
    do
        echo "***********************************************************************************"
        echo ${dev_paths[i]}
        echo ${model_paths[j]}
        echo ${limit}
        python evaluate.py \
            --model_path ${model_paths[j]} \
            --test_path ${dev_paths[i]}  \
            --batch_size 16 \
            --max_seq_len 512 \
            --limit ${limit}
    done
done
