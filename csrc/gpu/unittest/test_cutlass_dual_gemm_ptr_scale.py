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
from paddle.incubate.nn.functional import fused_bias_act
from paddlenlp_ops import (
    cutlass_fp8_fp8_fp8_dual_gemm_fused_scale_ptr as fp8_dual_gemm_fused_ptr_scale,
)

paddle.seed(1)


def create_random_cuda_tensor(shape, dtype, mean: float = 0, std: float = 1):
    return paddle.empty(shape, dtype=dtype).normal_(mean, std)


class CutlassBlockGemmTest(unittest.TestCase):
    def native_dual_gemm(
        self,
        x,
        w_fused,
        x_scale,
        w_scale,
        transpose_x=False,
        transpose_y=True,
        bias0=None,
        bias1=None,
        scale0=1.0,
        scale1=1.0,
        scale_out=1.0,
        act="swiglu",
        output_dtype="bfloat16",
    ):

        x = x.cast(paddle.float32)
        w_fused = w_fused.cast(paddle.float32).t()
        n = w_fused.shape[-1] // 2
        temp_out = paddle.matmul(x, w_fused)
        temp_out[:, :n] = temp_out[:, :n] * x_scale * w_scale[0, 0]
        temp_out[:, n:] = temp_out[:, n:] * x_scale * w_scale[1, 0]
        out = fused_bias_act(temp_out, None, act_method=act)
        out = out * scale_out
        return out.cast(output_dtype)

    def test_cutlass_fp8_dual_gemm_ptr_scale(self):
        M = 32
        N = 18944
        K = 3584
        x = (
            create_random_cuda_tensor([M, K], "float32", mean=0, std=0.1)
            .clip(min=-448.0, max=448.0)
            .cast(paddle.float8_e4m3fn)
        )
        w_0 = create_random_cuda_tensor([N, K], "float32", mean=0, std=0.1).clip(min=-448.0, max=448.0)
        w_1 = create_random_cuda_tensor([N, K], "float32", mean=0, std=0.1).clip(min=-448.0, max=448.0)
        w_fuesd = paddle.concat([w_0, w_1], axis=0).cast(paddle.float8_e4m3fn)
        x_scale = 0.03
        w_scale0 = 0.04
        w_scale1 = 0.05
        scale_out = 0.06
        x_scale = paddle.to_tensor(x_scale)
        w_scale = paddle.to_tensor([[w_scale0], [w_scale1]])

        ref_out = self.native_dual_gemm(
            x,
            w_fuesd,
            x_scale,
            w_scale,
            transpose_x=False,
            transpose_y=True,
            bias0=None,
            bias1=None,
            scale0=x_scale * w_scale0,
            scale1=x_scale * w_scale1,
            scale_out=scale_out,
            act="swiglu",
            output_dtype="bfloat16",
        )

        out = fp8_dual_gemm_fused_ptr_scale(
            x,
            w_fuesd,
            x_scale,
            w_scale,
            transpose_x=False,
            transpose_y=True,
            bias0=None,
            bias1=None,
            scale0=x_scale * w_scale0,
            scale1=x_scale * w_scale1,
            scale_out=scale_out,
            act="swiglu",
            output_dtype="bfloat16",
        )

        np.testing.assert_allclose(
            ref_out.cast(paddle.float32).numpy(), out.cast(paddle.float32).numpy(), rtol=1e-3, atol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
