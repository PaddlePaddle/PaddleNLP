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
 # This script is used to train lstm models
###

unset CUDA_VISIBLE_DEVICES
LANGUAGE=en

if [[ $LANGUAGE == "ch" ]]; then
    VOCAB_PATH=vocab.char
elif [[ $LANGUAGE == "en" ]]; then
    VOCAB_PATH=vocab_QQP
fi

python -m paddle.distributed.launch --gpus "5" train.py \
    --device=gpu \
    --lr=4e-4 \
    --batch_size=64 \
    --epochs=12 \
    --vocab_path=$VOCAB_PATH   \
    --language=$LANGUAGE \
    --save_dir="./checkpoints_"${LANGUAGE}
