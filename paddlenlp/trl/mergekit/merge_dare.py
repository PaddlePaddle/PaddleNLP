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


class MergeDare:
    def __init__(self, merge_config):
        self.merge_config = merge_config

    def merge_op(self, v_0, v_1, k=0.9):
        """
        Ties between two task vectors.
        """
        # Pruning
        v_0_processed, mask_0 = self.apply_bernoulli_mask(v_0, k)
        v_1_processed, mask_1 = self.apply_bernoulli_mask(v_1, k)

        assert v_0.shape == v_1.shape
        print("v_0_processed:")
        print(v_0_processed)
        print("v_1_processed:")
        print(v_1_processed)
        print("mask_0:")
        print(mask_0)
        print("mask_1:")
        print(mask_1)
        # 创建联合掩码：只有两个掩码都为 1 时，表示两者都有效
        joint_mask = (mask_0 == 1) & (mask_1 == 1)

        v_merge = v_0_processed + v_1_processed
        # 两者都有效时，取平均
        v_merge[joint_mask] = v_merge[joint_mask] * 0.5
        v_merge = self.merge_config.linear_ratio * v_merge
        return v_merge

    def merge_dict(self, dict1, dict2):
        if dict1.keys() != dict2.keys():
            raise ValueError(f"The keys of two dictionaries are different. {dict1.keys()} ！= {dict2.keys()} ")
        dict_merge = {}
        for key in dict1:
            dict_merge[key] = self.merge_op(dict1[key], dict2[key])
        return dict_merge

    def apply_bernoulli_mask(self, delta_t, p):
        # 从伯努利分布采样 m^t
        m_t = np.random.binomial(1, p, size=delta_t.shape).astype(delta_t.dtype)
        print("tensor:")
        print(delta_t)
        print("bernoulli mask：")
        print(m_t)
        # 计算 (1 - m^t) ⊙ δ^t
        delta_t_tilde = (1 - m_t) * delta_t
        print("delta_t_tilde:")
        print(delta_t_tilde)
        # 归一化 δ̃^t / (1 - p)
        delta_t_hat = delta_t_tilde / (1 - p)
        print("delta_t_hat归一化:")
        print(delta_t_hat)
        return delta_t_hat, 1 - m_t


class MergeConfig:
    def __init__(self, linear_ratio):
        self.linear_ratio = linear_ratio


merge_config = MergeConfig(linear_ratio=1)
merger = MergeDare(merge_config)

# 示例张量
v_0 = np.array([[1, 2, 3, 4, 5, 6, -7, -8, 9, 10], [3, 2, 4, 2.0, 1, 4, 5, 6, 4, 3]])
v_1 = np.array([[3, 2, 7, 2, 1.0, 2, 2, 3, 4, 4], [1, 5, 3, 6, 5, 2, -7, -8, 1, 10]])

# 合并操作
merged_vector = merger.merge_op(v_0, v_1, k=0.2)
print("Merged Vector:")
print(merged_vector)
