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
import numpy as np


class SparsificationMethod:
    def __init__(self, merge_config):
        self.merge_config = merge_config

    def sparsify_dare(self, v0, v1, drop_rate):
        v0, mask0 = self.apply_bernoulli_mask(v0, drop_rate)
        v1, mask1 = self.apply_bernoulli_mask(v1, drop_rate)

        return v0, v1, mask0, mask1

    def apply_bernoulli_mask(self, delta_t, p):
        # m^t
        m_t = np.random.binomial(1, p, size=delta_t.shape).astype(delta_t.dtype)
        # caculate (1 - m^t) ⊙ δ^t
        delta_t_tilde = (1 - m_t) * delta_t
        # δ̃^t / (1 - p)
        delta_t_hat = delta_t_tilde / (1 - p)
        return delta_t_hat, 1 - m_t

    def sparsify_della(self, v0, v1, drop_rate):
        v0, mask0 = self.magprune(v0, drop_rate)
        v1, mask1 = self.magprune(v1, drop_rate)

        return v0, v1, mask0, mask1

    def magprune(self, delta, p, epsilon):
        if np.all(delta == 0):
            return np.zeros_like(delta)
        # 1: ranking
        # abs
        abs_tensor = np.abs(delta)
        # descent order
        sorted_indices_flat = np.argsort(-abs_tensor.flatten())
        # ranking
        ranks_flat = np.empty_like(sorted_indices_flat)
        ranks_flat[sorted_indices_flat] = np.arange(1, delta.size + 1)
        # reshape original shape
        ranks = ranks_flat.reshape(delta.shape)
        # 2: caclculate drop rate p_i
        n = np.size(delta)
        delta_p = ranks * epsilon / n  # Δ_i =  ε/n * r_i
        p_min = p - epsilon / 2  # minimal drop rate
        p_i = p_min + delta_p  # p_i for each parameter
        p_i = np.clip(p_i, 0, 1)  # garantee that probability stays within [0, 1]
        # 3: drop parameters according to their probabilities

        m_i = np.random.binomial(1, p_i)
        retained_mask = m_i == 0  # mask for retained parameters
        adjusted_delta = delta * retained_mask
        adjusted_delta = adjusted_delta / (1 - p_i)
        return adjusted_delta, retained_mask
