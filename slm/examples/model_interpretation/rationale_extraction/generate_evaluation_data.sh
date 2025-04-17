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

###
 # This script concatenates results from previous running to generate a formated result for evaluation use
### 

BASE_MODEL=$1
INTER_MODE=$2
LANGUAGE=$3
TASK=$4

PRED_PATH=../task/${TASK}/output/${TASK}_${LANGUAGE}.${BASE_MODEL}/interpret.${INTER_MODE}
SAVE_PATH=./evaluation_data/${TASK}/${BASE_MODEL}_${INTER_MODE}_${LANGUAGE}

SAVE_DIR=./evaluation_data/${TASK}/
[ -d $SAVE_DIR ] || mkdir -p $SAVE_DIR

python3 generate_evaluation_data.py \
    --data_dir ./prediction/${TASK}/${BASE_MODEL}_${INTER_MODE}_${LANGUAGE} \
    --data_dir2 ./rationale/${TASK}/${BASE_MODEL}_${INTER_MODE}_${LANGUAGE} \
    --pred_path $PRED_PATH \
    --save_path $SAVE_PATH \
    --inter_mode $INTER_MODE \
    --base_model $BASE_MODEL \
    --language $LANGUAGE