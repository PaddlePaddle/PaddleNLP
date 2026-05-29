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


class MergeTies:
    def __init__(self, merge_config):
        self.merge_config = merge_config

    def merge_op(self, v0, v1, k=0.9):
        """
        Ties between two task vectors.
        """
        # Pruning
        v0 = self.pruning(v0, k)
        v1 = self.pruning(v1, k)
        print(v0)
        print(v1)

        assert v0.shape == v1.shape
        # 比较符号是否不同：np.sign 返回元素的符号（+1, -1 或 0）
        sign_diff = np.sign(v0) != np.sign(v1)

        # 计算绝对值
        abs_v0 = np.abs(v0)
        abs_v1 = np.abs(v1)

        # 找到绝对值较小的元素的位置
        v0_smaller = abs_v0 <= abs_v1

        # 创建掩码：符号不同且绝对值较小的元素置为 0
        # 当绝对值相同符号不同时与v1取一致
        v0_mask = ~(sign_diff & v0_smaller)
        v1_mask = ~(sign_diff & ~v0_smaller)
        print(v0_mask)
        print(v1_mask)
        # 置零处理
        v0_processed = v0 * v0_mask.astype(v0.dtype)
        v1_processed = v1 * v1_mask.astype(v1.dtype)
        print(v0_processed)
        print(v1_processed)
        # 求平均
        # result = (v0_processed + v1_processed) / 2
        # 按比例求task vector
        joint_mask = v0_mask & v1_mask

        # 计算平均值和直接相加的结果
        avg_result = (v0_processed + v1_processed) / 2  # 只用于都为 True 的情况
        sum_result = v0_processed + v1_processed  # 用于有0的情况

        # 根据联合掩码选择结果
        v_merge = np.where(joint_mask, avg_result, sum_result)

        # 按比例调整
        v_merge = self.merge_config.linear_ratio * v_merge

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
        print("阈值：{}".format(threshold))
        mask = abs_v0 > threshold  # Create a mask for elements above the threshold
        flat_v0 = flat_v0 * mask.astype(flat_v0.dtype)  # Apply the mask
        return flat_v0.reshape(v0.shape)  # Reshape back to the original shape


class MergeConfig:
    def __init__(self, linear_ratio):
        self.linear_ratio = linear_ratio


merge_config = MergeConfig(linear_ratio=0.8)
merger = MergeTies(merge_config)

# 示例张量
v_0 = np.array([[1, 2, 3, 4, 5, 6, -7, -8, 9, 10], [3, 2, 4, 2.0, 1, 4, 5, 6, 4, 3]])
v_1 = np.array([[3, 2, 7, 2, 1.0, 2, 2, 3, 4, 4], [1, 5, 3, 6, 5, 2, -7, -8, 1, 10]])

# 合并操作
merged_vector = merger.merge_op(v_0, v_1, k=0.8)
print("Merged Vector:", merged_vector)
