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
from paddle.incubate.nn.functional import swiglu
from paddlenlp_ops import cutlass_fp8_fp8_fp8_dual_gemm_fused

A = paddle.ones([2, 32, 64], dtype="float8_e4m3fn")
B0 = paddle.ones([2, 128, 64], dtype="float8_e4m3fn")
B1 = paddle.ones([2, 128, 64], dtype="float8_e4m3fn")

result = cutlass_fp8_fp8_fp8_dual_gemm_fused(
    A,
    B0,
    B1,
    bias0=None,
    bias1=None,
    transpose_x=False,
    transpose_y=True,
    scale0=0.1,
    scale1=0.1,
    scale_out=0.5,
    act="swiglu",
)


A = paddle.ones([2, 32, 64], dtype="float32")
B0 = paddle.ones([2, 128, 64], dtype="float32")
B1 = paddle.ones([2, 128, 64], dtype="float32")
tem0 = 0.1 * paddle.matmul(A, B0.transpose([0, 2, 1]))
tem1 = 0.1 * paddle.matmul(A, B1.transpose([0, 2, 1]))
expect_result = 0.5 * swiglu(tem0, tem1)


print("result: ", result)
print("expect_result: ", expect_result)
