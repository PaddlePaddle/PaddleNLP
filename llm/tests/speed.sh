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

export test_ci_no_save_model=1
# PYTHONPATH=../ python3.9 -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./qwen/pretrain_argument_stage2.json
# PYTHONPATH=../ python3.9 -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./qwen/pretrain_qwen-7b-tp2sd4-stage2.json
# PYTHONPATH=../ python3.9 -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./llama/pretrain_baichuan2-13b-tp4sd2-stage2.json
PYTHONPATH=../ python3.9 -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./llama/pretrain_baichuan2-13b-sd8-stage2.json
# PYTHONPATH=../ python3.9 -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./llama/pretrain_baichuan2-13b-tp2sd4-stage2.json
# PYTHONPATH=../ python3.9 -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./llama/pretrain_baichuan2-7b-tp2sd4-stage2.json
# PYTHONPATH=../ python3.9 -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./llama/pretrain_flagalpha-llama2-13b-tp2sd4-stage2.json
# PYTHONPATH=../ python3.9 -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./llama/pretrain_flagalpha-llama2-7b-tp2sd4-stage2.json
# PYTHONPATH=../ python3.9 -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./llama/pretrain_linly-llama2-7b-tp2sd4-stage2.json
# PYTHONPATH=../ python3.9 -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./llama/pretrain_llama-13b-tp2sd4-stage2.json
# PYTHONPATH=../ python3.9 -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./llama/pretrain_llama2-13b-tp2sd4-stage2.json
# PYTHONPATH=../ python3.9 -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./llama/pretrain_llama2-7b-tp2sd4-stage2.json
# PYTHONPATH=../ python3.9 -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./llama/pretrain_llama-7b-tp2sd4-stage2.json
# PYTHONPATH=../ python3.9 -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./llama/pretrain_ziya-llama-13b-tp2sd4-stage2.json
   
