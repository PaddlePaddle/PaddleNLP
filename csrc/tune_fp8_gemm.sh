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

# llama2-7B
# nohup python ./gpu/test_fp8_gemm.py \
#         --m_min 32 \
#         --m_max 2049 \
#         --n 4096 12288 \
#         --k 4096 11008 \
#         >  tune_gemm.log 2>&1 &

# llama3-8B
nohup python ./gpu/test_fp8_gemm.py \
        --m_min 32 \
        --m_max 32768 \
        --n 4096 6144 \
        --k 4096 14336 \
        >  tune_gemm.log 2>&1 &