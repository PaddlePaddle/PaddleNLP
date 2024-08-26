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

import paddle
from paddlenlp_ops import tune_cublaslt_gemm

M_tensor = paddle.to_tensor([32768])

# llama3.1-8b
k1 = [4096, 4096, 4096, 14336]
n1 = [6144, 4096, 28672, 4096]

# llama3.1-405b mp=8
k2 = [16384, 16384, 16384, 6656]
n2 = [2560, 16384, 13312, 16384]

# qwen2-1.5b
k3 = [1536, 1536, 1536, 8960]
n3 = [2048, 1536, 17920, 1536]

# qwen2-7b
k4 = [3584, 3584, 3584, 18944]
n4 = [4608, 3584, 37888, 3584]

K_tensor = paddle.to_tensor(k1 + k2 + k3 + k4)
N_tensor = paddle.to_tensor(n1 + n2 + n3 + n4)

Dtype = "int8"
Path = "./cublaslt_gemm_search.csv"

tune_cublaslt_gemm(M_tensor, K_tensor, N_tensor, Dtype, True, False, Path)
