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


class SparsifyMethod:
    def __init__(self, merge_config):
        self.merge_config = merge_config

    def sparsify(self, tensor):
        if self.merge_config.sparsify_type is None:
            return tensor
        if self.merge_config.sparsify_type == "dare":
            return self.dare(tensor)
        elif self.merge_config.sparsify_type == "magprune":
            return self.magprune(tensor)
        elif self.merge_config.sparsify_type == "trim":
            return self.trim(tensor)
        else:
            raise ValueError(f"Unknown sparsify method: {self.merge_config.sparsify_type}")

    def dare(self, tensor):
        if self.merge_config.tensor_type == "np":
            mask = np.random.binomial(1, self.merge_config.reserve_p, size=tensor.shape).astype(tensor.dtype)
            tensor *= mask
            if self.merge_config.rescale:
                tensor /= self.merge_config.reserve_p
            return tensor
        else:
            raise NotImplementedError("Paddle Tensor is not supported yet.")

    def magprune(self, tensor):
        if self.merge_config.tensor_type == "np":
            if np.all(tensor == 0):
                return tensor
            drop_p = 1 - self.merge_config.reserve_p
            # 1: ranking(descending)
            abs_tensor = np.abs(tensor)
            sorted_indices = np.argsort(-abs_tensor.flatten())

            # 2: caclculate drop rate p_i
            probs = np.empty_like(sorted_indices)
            probs[sorted_indices] = np.arange(tensor.size).astype(tensor.dtype)
            probs = probs.reshape(tensor.shape)  # r_i ∈ {0，1，... ,n}
            probs = probs * self.merge_config.epsilon / tensor.size  # Δ_i =  ε/n * r_i
            p_min = drop_p - self.merge_config.epsilon / 2  # minimal drop rate
            probs += p_min  # p_i for each parameter

            # 3: drop parameters according to their probabilities
            mask = np.random.binomial(1, probs)
            tensor *= (1 - mask).astype(tensor.dtype)
            if self.merge_config.rescale:
                tensor /= 1 - probs
            return tensor
        else:
            raise NotImplementedError("Paddle Tensor is not supported yet.")

    def trim(self, tensor):
        if self.merge_config.tensor_type == "np":
            shape = tensor.shape
            org_sum = np.sum(np.abs(tensor))
            tensor = tensor.flatten()
            abs_tensor = np.abs(tensor)
            threshold = np.quantile(abs_tensor, 1 - self.merge_config.reserve_p)

            if self.merge_config.rescale:
                org_sum = np.sum(np.abs(tensor))
                tensor[abs_tensor < threshold] = 0
                new_sum = np.sum(np.abs(tensor))
                if org_sum >= 1e-8 and new_sum >= 1e-8:
                    tensor *= org_sum / new_sum
            else:
                tensor[abs_tensor < threshold] = 0
            return tensor.reshape(shape)
        else:
            raise NotImplementedError("Paddle Tensor is not supported yet.")
