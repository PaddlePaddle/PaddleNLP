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

# 指定GPU
export CUDA_VISIBLE_DEVICES=0
# 开启FP8 cutlass调优(必要性配置)
export FLAGS_use_cutlass_device_best_config_path=tune

# 根据不同模型，调整gemm参数，其中n和k的的值需要成对出现，具体逻辑可参考相关.py文件。
# llama3-8B 单GEMM调优参数设置
nohup python ./utils/tune_cutlass_fp8_gemm.py \
        --m_min 32 \
        --m_max 32768 \
        --n 6144 4096 4096 \
        --k 4096 4096 14336 \
        >  tune_fp8_gemm.log 2>&1 &

# llama3-8B 双GEMM调优参数设置
# nohup python ./utils/tune_cutlass_fp8_dual_gemm.py \
#         --m_min 32 \
#         --m_max 32768 \
#         --n 14336 \
#         --k 4096 \
#         >  tune_fp8_dual_gemm.log 2>&1 &
