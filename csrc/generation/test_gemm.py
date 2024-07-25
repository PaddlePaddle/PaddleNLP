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

from paddlenlp_ops import tune_cublaslt_gemm
import paddle

K = []
N = []

# Shape initialization
K.extend([1024, 2048])
N.extend([4096, 8192])

M_tensor = paddle.to_tensor([1024])
K_tensor = paddle.to_tensor(K)
N_tensor = paddle.to_tensor(N)

Dtype = "int8"
Path = "./search2.csv"

tune_cublaslt_gemm(M_tensor, K_tensor, N_tensor, Dtype, Path)
