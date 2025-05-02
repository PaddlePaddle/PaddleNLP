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
from paddlenlp_ops import per_tensor_quant_fp8

paddle.seed(1)


def create_random_cuda_tensor(shape, dtype, mean: float = 0, std: float = 1):
    return paddle.empty(shape, dtype=dtype).normal_(mean, std)


class DynamicPerTensorQuantFP8Test(unittest.TestCase):
    def native_per_tensor_quant_fp8(self, x, scale=None):
        x_fp32 = x.cast("float32")
        if scale is None:
            x_s = x_fp32.abs().max() / 448.0
        else:
            x_s = scale
        x_q = x_fp32 / x_s
        x_q = x_q.clip(min=-448.0, max=448.0)
        return x_q.cast("float8_e4m3fn"), x_s.reshape([1])

    def test_dynamic_per_tensor_quant_fp8_bf16(self):
        num_tokens = 1024
        hidden_size = 1152
        dtype = paddle.bfloat16
        x = paddle.rand([num_tokens, hidden_size], dtype=dtype)
        ref_out, ref_scale = self.native_per_tensor_quant_fp8(x)
        out, scale = per_tensor_quant_fp8(x, None)

        np.testing.assert_allclose(ref_out.numpy(), out.numpy(), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(ref_scale.numpy(), scale.numpy(), rtol=1e-3, atol=1e-3)

    def test_dynamic_per_tensor_quant_fp8_fp32(self):
        num_tokens = 1024
        hidden_size = 1152
        dtype = paddle.float32
        x = paddle.rand([num_tokens, hidden_size], dtype=dtype)

        ref_out, ref_scale = self.native_per_tensor_quant_fp8(x)
        out, scale = per_tensor_quant_fp8(x, None)

        np.testing.assert_allclose(ref_out.numpy(), out.numpy(), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(ref_scale.numpy(), scale.numpy(), rtol=1e-3, atol=1e-3)

    def test_dynamic_per_tensor_quant_fp8_fp16(self):
        num_tokens = 1024
        hidden_size = 1152
        dtype = paddle.float16
        x = paddle.rand([num_tokens, hidden_size], dtype=dtype)

        ref_out, ref_scale = self.native_per_tensor_quant_fp8(x)
        out, scale = per_tensor_quant_fp8(x, None)

        np.testing.assert_allclose(ref_out.numpy(), out.numpy(), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(ref_scale.numpy(), scale.numpy(), rtol=1e-3, atol=1e-3)

    def test_static_per_tensor_quant_fp8_bf16(self):
        num_tokens = 1024
        hidden_size = 1152
        dtype = paddle.bfloat16
        x = paddle.rand([num_tokens, hidden_size], dtype=dtype)
        scale = paddle.rand([1], dtype=paddle.float32).clip(0.0001)
        ref_out, ref_scale = self.native_per_tensor_quant_fp8(x, scale)
        out, scale = per_tensor_quant_fp8(x, scale)

        np.testing.assert_allclose(ref_out.numpy(), out.numpy(), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(ref_scale.numpy(), scale.numpy(), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
