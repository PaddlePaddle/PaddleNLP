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

from paddlenlp.utils import infohub


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


def random_hadamard_matrix(size, dtype, quantization_config):
    if not quantization_config.hadamard_is_block:
        A = paddle.randint(low=0, high=2, shape=[size, size]).astype("float32") * 2 - 1
        Q, _ = paddle.linalg.qr(A)
        return Q.astype(dtype), 1
    else:
        if quantization_config.hadamard_block_size != -1:
            assert size % quantization_config.hadamard_block_size == 0, "Please choose a correct block_size"
            num_blocks = size // quantization_config.hadamard_block_size
            Q = paddle.diag(paddle.ones((quantization_config.hadamard_block_size,), dtype="float32"))
            block = matmul_hadU(Q)
            return block, quantization_config.hadamard_block_size
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


def hadamard_matmul(input, side, hadamard_maxtrix, block_size):
    # left -> H.T@input right -> input@H
    origin_shape = input.shape
    input = input.reshape([-1, origin_shape[-1]])
    if side == "left":
        # H.T@input -> (input.T@H).T
        input = input.transpose([1, 0])
    block_num = input.shape[-1] // block_size
    output = input.reshape([-1, block_num, block_size]) @ hadamard_maxtrix
    output = output.reshape([-1, block_num * block_size])
    if side == "left":
        output = output.transpose([1, 0])
    output = output.reshape(origin_shape)

    return output


def apply_hadamard_matmul(x, side, quantization_config=None, dequant=False):
    if getattr(infohub, "hadamard") is None:
        setattr(infohub, "hadamard", {})
    if side == "left":
        x_shape = x.shape[0]
    else:
        x_shape = x.shape[-1]
    if x_shape in infohub.hadamard:
        hadamard_maxtrix, block_size = infohub.hadamard[x_shape]
    else:
        hadamard_matrix, block_size = random_hadamard_matrix(x_shape, x.dtype, quantization_config)
        infohub.hadamard[x_shape] = (hadamard_matrix, block_size)
    if block_size > 1:
        target_x = hadamard_matmul(x, side, hadamard_maxtrix, block_size)
    else:
        if dequant:
            hadamard_matrix = hadamard_matrix.T
        if side == "right":
            target_x = x @ hadamard_matrix
        else:
            target_x = hadamard_matrix.T @ x

    return target_x, block_size
