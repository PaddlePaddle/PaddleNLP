# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


def matmul_hadU(X):

    input = X.clone().reshape((-1, X.shape[-1], 1))
    output = input.clone()
    while input.shape[1] > 1:
        input = input.reshape((input.shape[0], input.shape[1] // 2, 2, input.shape[2]))
        output = output.reshape(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.reshape((input.shape[0], input.shape[1], -1))
        (input, output) = (output, input)
    del output

    return input.reshape(X.shape)


def random_hadamard_matrix(size, dtype, is_block=False):
    if not is_block:
        A = paddle.randint(low=0, high=2, shape=[size, size]).astype("float32") * 2 - 1
        Q, _ = paddle.linalg.qr(A)
        return Q.astype(dtype), 1
    else:
        num_blocks = size
        while not (num_blocks % 2):
            num_blocks = num_blocks // 2
        block_size = size // num_blocks
        Q = paddle.diag(paddle.ones((block_size,), dtype="float32"))
        block = matmul_hadU(Q)
        large_matrix = paddle.zeros([size, size])

        for i in range(num_blocks):
            start_row = i * block_size
            start_col = i * block_size
            large_matrix[start_row : start_row + block_size, start_col : start_col + block_size] = block
    return large_matrix.cast(dtype), block_size
