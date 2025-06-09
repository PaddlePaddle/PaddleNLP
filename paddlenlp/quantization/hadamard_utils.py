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
    if quantization_config.hadamard_block_size < 0:
        A = paddle.randint(low=0, high=2, shape=[size, size]).astype("float32") * 2 - 1
        Q, _ = paddle.linalg.qr(A)
        return Q.astype(dtype), 1
    else:
        assert size % quantization_config.hadamard_block_size == 0, "Please choose a correct block_size"
        Q = paddle.diag(paddle.ones((quantization_config.hadamard_block_size,), dtype="float32"))
        block = matmul_hadU(Q)
        print("random_hadamard_matrix", block, quantization_config.hadamard_block_size)
        return block, quantization_config.hadamard_block_size


def hadamard_matmul(input, side, hadamard_matrix, block_size):
    # left -> H.T@input right -> input@H
    origin_shape = input.shape
    input = input.reshape([-1, origin_shape[-1]])
    if side == "left":
        # H.T@input -> (input.T@H).T
        input = input.transpose([1, 0])
    block_num = input.shape[-1] // block_size
    output = input.reshape([-1, block_num, block_size]) @ hadamard_matrix
    output = output.reshape([-1, block_num * block_size])
    if side == "left":
        output = output.transpose([1, 0])
    output = output.reshape(origin_shape)

    return output


def apply_hadamard_matmul(x, side, quantization_config=None, dequant=False):
    if getattr(infohub, "hadamard") is None:
        setattr(infohub, "hadamard", {})

    if quantization_config.hadamard_block_size < 0:
        if side == "left":
            block_size = x.shape[0]
        else:
            block_size = x.shape[-1]
    else:
        block_size = quantization_config.hadamard_block_size

    if block_size in infohub.hadamard:
        hadamard_matrix, hadamard_scale = infohub.hadamard[block_size]
    else:
        hadamard_matrix, hadamard_scale = random_hadamard_matrix(block_size, x.dtype, quantization_config)
        infohub.hadamard[block_size] = (hadamard_matrix, hadamard_scale)

    if hadamard_scale > 1:
        target_x = hadamard_matmul(x, side, hadamard_matrix, block_size)
    else:
        if dequant:
            hadamard_matrix = hadamard_matrix.T
        if side == "right":
            target_x = x @ hadamard_matrix
        else:
            target_x = hadamard_matrix.T @ x

    return target_x, hadamard_scale
