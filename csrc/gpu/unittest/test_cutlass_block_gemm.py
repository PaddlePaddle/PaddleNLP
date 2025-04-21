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

import unittest

import numpy as np
import paddle
from paddlenlp_ops import cutlass_fp8_fp8_half_block_gemm_fused as fp8_block_gemm_fused

paddle.seed(1)


def create_random_cuda_tensor(shape, dtype, mean: float = 0, std: float = 1):
    return paddle.empty(shape, dtype=dtype).normal_(mean, std)


class CutlassBlockGemmTest(unittest.TestCase):
    def native_w8a8_block_fp8_matmul(self, A, B, As, Bs, block_size, output_dtype="float16"):
        """This function performs matrix multiplication with block-wise quantization using native paddle.

        It takes two input tensors `A` and `B` with scales `As` and `Bs`.
        The output is returned in the specified `output_dtype`.
        """

        A = A.cast(paddle.float32)
        B = B.cast(paddle.float32)
        assert A.shape[-1] == B.shape[-1]
        assert len(block_size) == 2
        block_n, block_k = block_size[0], block_size[1]
        assert (A.shape[-1] + block_k - 1) // block_k == As.shape[-1]
        assert A.shape[:-1] == As.shape[:-1]

        M = A.numel() // A.shape[-1]
        N, K = B.shape
        origin_C_shape = A.shape[:-1] + [N]
        A = A.reshape([M, A.shape[-1]])
        As = As.reshape([M, As.shape[-1]])
        n_tiles = (N + block_n - 1) // block_n
        k_tiles = (K + block_k - 1) // block_k
        assert n_tiles == Bs.shape[0]
        assert k_tiles == Bs.shape[1]

        C_shape = [M, N]
        C = paddle.zeros(C_shape, dtype=paddle.float32)
        A_tiles = [A[:, i * block_k : min((i + 1) * block_k, K)] for i in range(k_tiles)]
        B_tiles = [
            [
                B[
                    j * block_n : min((j + 1) * block_n, N),
                    i * block_k : min((i + 1) * block_k, K),
                ]
                for i in range(k_tiles)
            ]
            for j in range(n_tiles)
        ]
        C_tiles = [C[:, j * block_n : min((j + 1) * block_n, N)] for j in range(n_tiles)]
        As_tiles = [As[:, i : i + 1] for i in range(k_tiles)]

        for i in range(k_tiles):
            for j in range(n_tiles):
                a = A_tiles[i]
                b = B_tiles[j][i]
                c = C_tiles[j]
                s = As_tiles[i] * Bs[j][i]
                c[:, :] += paddle.matmul(a, b.t()) * s
        C = C.reshape(origin_C_shape).cast(output_dtype)
        return C

    def test_cutlass_fp8_block_gemm_fused(self):
        M = 32
        K = 7168
        N = 1536
        block_size = [128, 128]
        output_dtype = "float16"

        scale_k = (K + 128 - 1) // 128
        scale_n = (N + 128 - 1) // 128
        A = (
            create_random_cuda_tensor([M, K], "float32", mean=0, std=0.1)
            .clip(min=-448.0, max=448.0)
            .cast(paddle.float8_e4m3fn)
        )
        B = (
            create_random_cuda_tensor([N, K], "float32", mean=0, std=0.1)
            .clip(min=-448.0, max=448.0)
            .cast(paddle.float8_e4m3fn)
        )
        As = paddle.rand([M, scale_k], dtype=paddle.float32).clip(min=0.001)
        Bs = paddle.rand([scale_n, scale_k], dtype=paddle.float32).clip(min=0.001)

        ref_out = self.native_w8a8_block_fp8_matmul(A, B, As, Bs, block_size, output_dtype)
        out = fp8_block_gemm_fused(
            A,
            B,
            As.t(),
            Bs,
            bias=None,
            transpose_x=False,
            transpose_y=True,
            output_dtype=output_dtype,
            act="identity",
        )

        np.testing.assert_allclose(ref_out.numpy(), out.numpy(), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
