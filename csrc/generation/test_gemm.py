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

M = []
K = []
N = []

# Shape initialization
M.extend(range(1, 4, 1))
M.extend(range(4, 16, 4))
M.extend(range(16, 64, 16))
M.extend(range(64, 256, 32))
M.extend(range(256, 512, 64))
M.extend(range(512, 1024, 128))
M.extend(range(1024, 8193, 1024))

K.extend([1024, 2048])
N.extend([4096, 8192])

M_tensor = paddle.to_tensor(M)
K_tensor = paddle.to_tensor(K)
N_tensor = paddle.to_tensor(N)

Dtype = "int8"
Path = "./search2.csv"

tune_cublaslt_gemm(M_tensor, K_tensor, N_tensor, Dtype, Path)
