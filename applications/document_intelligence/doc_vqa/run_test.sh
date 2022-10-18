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

QUESTION=$1

# Question: NFC咋开门

if [ $# != 1 ];then
    echo "USAGE: sh script/run_cross_encoder_test.sh \$QUESTION"
    exit 1
fi

# compute scores for QUESTION and OCR parsing results  with Rerank module
cd Rerank
bash run_test.sh ${QUESTION}
cd ..

# extraction answer for QUESTION from the top1 of rank
cd Extraction
bash run_test.sh ${QUESTION}
cd ..
