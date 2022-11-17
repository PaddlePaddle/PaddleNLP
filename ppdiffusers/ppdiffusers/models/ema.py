# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class LitEma(nn.Layer):
    """
    Exponential Moving Average (EMA) of model updates

    Parameters:
        model: The model architecture for apply EMA.
        decay: The exponential decay. Default 0.9999.
        use_num_updates: Whether to use number of updates when computing
            averages.
    """

    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.register_buffer('decay',
                             paddle.to_tensor(decay, dtype=paddle.float32))
        self.register_buffer(
            'num_updates',
            paddle.to_tensor(0, dtype=paddle.int64)
            if use_num_upates else paddle.to_tensor(-1, dtype=paddle.int64))

        for name, p in model.named_parameters():
            if not p.stop_gradient:
                #remove as '.'-character is not allowed in buffers
                s_name = name.replace('.', '')
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(s_name, p.clone().detach())

        self.collected_params = []

    def forward(self, model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay,
                        (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with paddle.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if not m_param[key].stop_gradient:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname].scale_(decay)
                    shadow_params[sname].add_(m_param[key] * one_minus_decay)
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if not m_param[key].stop_gradient:
                m_param[key].copy_(shadow_params[self.m_name2s_name[key]], True)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `paddle.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `paddle.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.copy_(c_param, True)
