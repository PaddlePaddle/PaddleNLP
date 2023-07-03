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

# prepare, do once
# git clone https://github.com/JunnYu/PaddleNLP -b add_sd_diffusers_benchmark
# cd PaddleNLP/tests
# # benchmark tools
# wget https://paddle-qa.bj.bcebos.com/benchmark/tools.tar.gz
# tar -zxvf tools.tar.gz
# export script_path=test_tipc/configs/stable_diffusion_model/train_infer_python.txt
# # prepare data
# bash test_tipc/prepare.sh $script_path benchmark_train


cd PaddleNLP/tests
export BENCHMARK_ROOT=$PWD/tools
export script_path=test_tipc/configs/stable_diffusion_model/train_infer_python.txt
export FLAG_XFORMERS=True
export FLAG_RECOMPUTE=True
export run_model_config_type=dynamic_bs10_fp32_DP_N1C1
bash test_tipc/benchmark_train.sh $script_path benchmark_train $run_model_config_type











































