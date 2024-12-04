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

from .merge_sparsify import SparsificationMethod


class MergeTies:
    def __init__(self, merge_config):
        self.merge_config = merge_config

    def merge_op(self, v0, v1, sparsify_method=None):
        """
        Ties between two task vectors.
        """
        # Pruning
        assert v0.shape == v1.shape
        drop_rate = self.merge_config.drop_rate
        if self.merge_config.sparsify_type is not None:
            sparsify = SparsificationMethod(self.merge_config)
            v0 = sparsify.sparsify_method(v0)
            v1 = sparsify.sparsify_method(v1)
        else:
            v0 = self.pruning(v0, drop_rate)
            v1 = self.pruning(v1, drop_rate)
        # np.sign （+1, -1 或 0）
        sign_diff = np.sign(v0) != np.sign(v1)

        # calculate the absolute value of each element
        abs_v0 = np.abs(v0)
        abs_v1 = np.abs(v1)

        # find the smaller one
        v0_smaller = abs_v0 <= abs_v1

        # create masks: when signs differ and the absolute value is smaller than other
        v0_mask = ~(sign_diff & v0_smaller)
        v1_mask = ~(sign_diff & ~v0_smaller)
        v0_processed = v0 * v0_mask.astype(v0.dtype)
        v1_processed = v1 * v1_mask.astype(v1.dtype)
        v_merge = (
            1 - self.merge_config.linear_ratio
        ) * v0_processed * v0_mask + self.merge_config.linear_ratio * v1_processed * v1_mask

        return v_merge

    def merge_dict(self, dict1, dict2):
        if dict1.keys() != dict2.keys():
            raise ValueError(f"The keys of two dictionaries are different. {dict1.keys()} ！= {dict2.keys()} ")
        dict_merge = {}
        for key in dict1:
            dict_merge[key] = self.merge_op(dict1[key], dict2[key])
        return dict_merge

    def pruning(self, v0, k):
        """
        Prunes elements of the input array based on their magnitudes.

        Args:
            v0 (np.ndarray): Input array to be pruned.
            k (float): Fraction of elements to retain based on magnitude.

        Returns:
            np.ndarray: Pruned array with retained elements.
        """
        flat_v0 = v0.flatten()  # Flatten the input array
        abs_v0 = np.abs(flat_v0)  # Compute the absolute values
        threshold = np.quantile(abs_v0, 1 - k)  # Determine the pruning threshold
        mask = abs_v0 > threshold  # Create a mask for elements above the threshold
        flat_v0 = flat_v0 * mask.astype(flat_v0.dtype)  # Apply the mask
        return flat_v0.reshape(v0.shape)  # Reshape back to the original shape
