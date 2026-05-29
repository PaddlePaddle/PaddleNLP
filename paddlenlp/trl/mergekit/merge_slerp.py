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


class MergeSlerp:
    def __init__(self, merge_config):
        self.merge_config = merge_config

    def merge_op(self, v0, v1, eps=1e-8, dot_threshold=0.995):
        """
        Spherical linear interpolation between two values.
        """
        # Copy the vectors to reuse them later
        v0_copy = np.copy(v0)
        v1_copy = np.copy(v1)
        # Normalize the vectors to get the directions and angles
        v0 = v0 / (np.linalg.norm(v0) + eps)
        v1 = v1 / (np.linalg.norm(v1) + eps)
        # Dot product with the normalized vectors (can't use np.dot in W)
        dot = np.sum(v0 * v1)

        # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
        # if np.abs(dot) > dot_threshold:
        #     res = lerp(v0_copy, v1_copy)
        #     return res

        # Calculate initial angle between v0 and v1
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)

        # Angle at timestep t
        theta_t = theta_0 * self.merge_config.linear_ratio
        sin_theta_t = np.sin(theta_t)

        # Finish the slerp algorithm
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v_merge = s0 * v0_copy + s1 * v1_copy
        return v_merge

    def merge_dict(self, dict1, dict2):
        if dict1.keys() != dict2.keys():
            raise ValueError(f"The keys of two dictionaries are different. {dict1.keys()} ！= {dict2.keys()} ")
        dict_merge = {}
        for key in dict1:
            dict_merge[key] = self.merge_op(dict1[key], dict2[key])
        return dict_merge


class MergeConfig:
    def __init__(self, linear_ratio):
        self.linear_ratio = linear_ratio


merge_config = MergeConfig(linear_ratio=0.8)
merger = MergeSlerp(merge_config)

# 示例张量
v_0 = np.array([[1, 2, 3, 4, 5, 6, -7, -8, 9, 10], [3, 2, 4, 2, 1, 4, 5, 6, 4, 3]])
v_1 = np.array([[3, 2, 4, 2, 1, 4, 5, 6, 4, 3], [1, 2, 3, 4, 5, 6, -7, -8, 9, 10]])

# 合并操作
merged_vector = merger.merge_op(v_0, v_1)
print("Merged Vector:", merged_vector)
