# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import torch
from apex.normalization.fused_layer_norm import fused_rms_norm_affine as rms_norm


class TestRmsNorm(unittest.TestCase):
    def create(self):
        width = 4096
        eps = 1e-6

        x_in = torch.randn([2, 1024, width]).cuda() + 0.02
        # x_in = paddle.ones([1, 1, width], dtype=paddle.get_default_dtype())
        # w = paddle.randn([1, 1024, width]) + 0.2
        # w = torch.normal(mean=0.2, std=0.02, shape=[width]).cuda()
        # w2 = torch.normal(mean=0.2, std=0.02, shape=[width]).cuda()
        # w2 = paddle.ones(shape=[width], dtype=paddle.get_default_dtype())  * 1
        # w2 = 1

        inputs = np.load("rms_inputs.npz")
        w = torch.Tensor(inputs["w"]).cuda()
        w2 = torch.Tensor(inputs["w2"]).cuda()
        x_in = torch.Tensor(inputs["x_in"]).cuda()

        x_in.requires_grad = True  # stop_gradient = False
        w.requires_grad = True  # stop_gradient = False

        # np.savez("rms_inputs.npz",x_in=x_in.numpy(), w=w.numpy(), w2=w2.numpy())

        return x_in, w, eps, w2

    def test_backward(self):
        x_in, w, eps, w2 = self.create()
        ret1 = rms_norm(x_in, w, torch.Size((4096,)), eps) * w2
        ret1.sum().backward()
        x_in_g_1 = x_in.grad.cpu().numpy()
        w_g_1 = w.grad.cpu().numpy()

        grads = np.load("rms_grads.npz")
        # np.savez("rms_grads.npz", x_in_g=x_in_g_2, w_g=w_g_2)

        # np.testing.assert_equal(ret1.numpy(), ret2.numpy())
        np.testing.assert_allclose(ret1.detach().cpu().numpy(), grads["ret"], rtol=1e-16)
        np.testing.assert_allclose(w_g_1, grads["w_g"], rtol=3e-14)
        np.testing.assert_allclose(x_in_g_1, grads["x_in_g"], rtol=1e-16)
