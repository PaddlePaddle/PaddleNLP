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

log_dir=log_345m_mp1
rm -rf $log_dir

DIRECTORY=./auto_infer
if [ ! -d "$DIRECTORY" ]; then
  echo "start download ckpt"
  wget https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M_FP16.tar.gz
  tar -zxvf GPT_345M_FP16.tar.gz
fi

python -m paddle.distributed.launch --log_dir $log_dir --devices "1" \
    ./tools/auto_export.py \
    -c ./ppfleetx/configs/nlp/gpt/auto/generation_gpt_345M_single_card.yaml \
    -o Engine.save_load.ckpt_dir=./pretrained/auto
