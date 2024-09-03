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

import os

import paddle

if os.getenv("FLAGS_CUTLASS_FP8_GEMM", "True") == "True":
    from paddlenlp_ops import cutlass_fp8_fp8_half_gemm_fused as fp8_gemm_fused
else:
    from paddle.linalg import fp8_fp8_half_gemm_fused as fp8_gemm_fused

A = paddle.ones([2, 32, 64], dtype="float8_e4m3fn")
B = paddle.ones([2, 32, 64], dtype="float8_e4m3fn")

res0 = fp8_gemm_fused(
    A,
    B,
    bias=None,
    transpose_x=False,
    transpose_y=True,
    output_dtype="float16",
    scale=0.5,
    act="identity",
)
print("res0: ", res0)

A = paddle.ones([2, 32, 64], dtype="float8_e4m3fn")
B = paddle.ones([2, 128, 64], dtype="float8_e4m3fn")

res1 = fp8_gemm_fused(
    A,
    B,
    bias=None,
    transpose_x=False,
    transpose_y=True,
    output_dtype="bfloat16",
    scale=0.5,
    act="identity",
)

A = paddle.ones([2, 32, 64], dtype="float32")
B = paddle.ones([2, 128, 64], dtype="float32")
expect_result = 0.5 * paddle.matmul(A, B.transpose([0, 2, 1]))

result0 = paddle.equal_all(
    paddle.cast(res0, "float32"),
    paddle.to_tensor(expect_result),
)

result1 = paddle.equal_all(
    paddle.cast(res1, "float32"),
    paddle.to_tensor(expect_result),
)

print("result0: ", result0)
print("result1: ", result1)
