# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import nn
import numpy as np


class MemN2N(nn.Layer):
    """
    End to End Memory Networks model

    reference paper: https://arxiv.org/pdf/1503.08895v5.pdf
    """

    def __init__(self, config):
        """
        Model initialization
        
        Args:
            config: model configuration, see config.yaml for more detail
        """
        super(MemN2N, self).__init__()
        self.nwords = config.nwords
        self.init_hid = config.init_hid
        self.init_std = config.init_std
        self.nhop = config.nhop
        self.edim = config.edim
        self.mem_size = config.mem_size
        self.lindim = config.lindim
        self.max_grad_norm = config.max_grad_norm
        self.batch_size = config.batch_size

        self.checkpoint_dir = config.checkpoint_dir

        normal_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Normal(std=self.init_std))
        self.A = nn.Embedding(self.nwords, self.edim, weight_attr=normal_attr)
        self.C = nn.Embedding(self.nwords, self.edim, weight_attr=normal_attr)

        # Temporal Encoding
        self.T_A = nn.Embedding(self.mem_size,
                                self.edim,
                                weight_attr=normal_attr)
        self.T_C = nn.Embedding(self.mem_size,
                                self.edim,
                                weight_attr=normal_attr)

        # Linear mapping for q
        self.H = nn.Linear(self.edim,
                           self.edim,
                           weight_attr=normal_attr,
                           bias_attr=False)

        # output mapping
        self.W = nn.Linear(self.edim,
                           self.nwords,
                           weight_attr=normal_attr,
                           bias_attr=False)

    def forward(self, data):
        """
        The shape of data is [batch_size, mem_size], and the content is the id of each word
        """
        q = np.ndarray([self.batch_size, self.edim], dtype=np.float32)
        q.fill(self.init_hid)
        q = paddle.to_tensor(q)

        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int64)
        for i in range(self.mem_size):
            time[:, i] = i
        time = paddle.to_tensor(time)

        for hop in range(self.nhop):
            A_in_c = self.A(data)  # [batch_size, mem_size, edim]
            A_in_t = self.T_A(time)  # [batch_size, mem_size, edim]
            A_in = paddle.add(A_in_c, A_in_t)  # [batch_size, mem_size, edim]

            q_in = q.reshape([-1, 1, self.edim])  # [batch, 1, edim]
            A_out3d = paddle.matmul(q_in, A_in,
                                    transpose_y=True)  # [batch, 1, mem_size]
            A_out2d = A_out3d.reshape([-1, self.mem_size])
            p = nn.functional.softmax(A_out2d)  # [batch, mem_size]

            C_in_c = self.C(data)
            C_in_t = self.T_C(time)
            C_in = paddle.add(C_in_c, C_in_t)  # [batch_size, mem_size, edim]

            p_3d = p.reshape([-1, 1, self.mem_size])  # [batch, 1, mem_size]
            C_out3d = paddle.matmul(p_3d, C_in)  # [batch, 1, edim]

            C_out2d = C_out3d.reshape([-1, self.edim])  # [batch, edim]

            # Linear mapping and addition
            q_mapped = self.H(q)
            q_out = paddle.add(C_out2d, q_mapped)

            if self.lindim == self.edim:
                q = q_out
            elif self.lindim == 0:
                q = nn.functional.relu(q_out)
            else:
                F = q_out[:, :self.lindim]
                G = q_out[:, self.lindim:]
                K = nn.functional.relu(G)
                q = paddle.concat([F, K], axis=-1)

        predict = self.W(q)
        return predict
