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


class MergeDella:
    def __init__(self, merge_config):
        self.merge_config = merge_config

    def merge_op(self, v_0, v_1, k=0.5):
        v_0_processed, mask_0 = self.magprune(v_0, k, 0.3)
        v_1_processed, mask_1 = self.magprune(v_1, k, 0.3)
        print("v_0_processed:")
        print(v_0_processed)
        print("v_1_processed:")
        print(v_1_processed)
        print("mask_0:")
        print(mask_0)
        print("mask_1:")
        print(mask_1)
        assert v_0.shape == v_1.shape
        # # 创建联合掩码：只有两个掩码有 false 时，表示
        joint_mask = (mask_0 == 1) & (mask_1 == 1)
        # print("joint_mask:")
        # print(joint_mask)
        # # 初始化 v_merge
        # v_merge = np.zeros_like(v_0)

        # # 两者都有效时，取平均
        # v_merge[joint_mask] = (v_0_processed[joint_mask] + v_1_processed[joint_mask]) * 0.5

        # # 至少一个无效时，直接相加
        # v_merge[~joint_mask] = (v_0_processed[~joint_mask] + v_1_processed[~joint_mask])
        # v_merge = self.merge_config.linear_ratio * v_merge
        # print("v_merge:")
        v_merge = v_0_processed + v_1_processed
        # 如果有舍弃的那就不取平均，都没舍弃需要取平均
        v_merge[joint_mask] = v_merge[joint_mask] * 0.5
        print("v_merge:")
        print(v_merge)
        return v_merge

    def merge_dict(self, dict1, dict2):
        if dict1.keys() != dict2.keys():
            raise ValueError(f"The keys of two dictionaries are different. {dict1.keys()} ！= {dict2.keys()} ")
        dict_merge = {}
        for key in dict1:
            dict_merge[key] = self.merge_op(dict1[key], dict2[key])
        return dict_merge

    def magprune(self, delta, p, epsilon):
        if np.all(delta == 0):
            return np.zeros_like(delta)
        # 步骤 1: 排序和分配排名
        # 绝对值
        abs_tensor = np.abs(delta)
        print("绝对值：\n", abs_tensor)

        # 展平后排序索引（绝对值降序）
        sorted_indices_flat = np.argsort(-abs_tensor.flatten())
        print("展平后排序索引（降序）：", sorted_indices_flat)

        # 排名
        ranks_flat = np.empty_like(sorted_indices_flat)
        ranks_flat[sorted_indices_flat] = np.arange(1, delta.size + 1)

        # 恢复原始形状
        ranks = ranks_flat.reshape(delta.shape)
        print(ranks)
        # 步骤 2: 计算丢弃概率 p_i
        n = np.size(delta)
        print("n是：")
        print(n)
        delta_p = ranks * epsilon / n  # Δ_i =  ε/n * r_i
        p_min = p - epsilon / 2  # 最小丢弃概率
        p_i = p_min + delta_p  # 计算每个参数的丢弃概率 p_i
        print("丢弃概率是：")
        print(p_i)
        p_i = np.clip(p_i, 0, 1)  # 保证概率在 [0, 1] 范围内
        print("最终的丢弃概率是：")
        print(p_i)
        # 步骤 3: 采样丢弃

        m_i = np.random.binomial(1, p_i)  # 采样得到 m_i (0 或 1)
        print("采样的结果是：")
        print(m_i)
        retained_mask = m_i == 0  # 保留参数的掩码
        print("保留的掩码是：")
        print(retained_mask)
        adjusted_delta = delta * retained_mask
        print("掩码后的delta是：")
        print(adjusted_delta)
        adjusted_delta = adjusted_delta / (1 - p_i)
        print("最终的delta是：")
        print(adjusted_delta)
        return adjusted_delta, retained_mask


class MergeConfig:
    def __init__(self, linear_ratio):
        self.linear_ratio = linear_ratio


merge_config = MergeConfig(linear_ratio=1)
merger = MergeDella(merge_config)

# 示例张量
v_0 = np.array([[1, 2, 3, 4, 5, 6.0, -7, -8, 9, 10], [3, 2, 4, 2, 1, 4, 5, 6, 4, 3]])
v_1 = np.array([[3, 2, 7, 3, 2, 3, 2, 3, 4, 3], [1, 5, 3, 6, 5.0, 2, -7, -8, 1, 10]])


# 合并操作
merged_vector = merger.merge_op(v_0, v_1, k=0.5)
print("Merged Vector:", merged_vector)
