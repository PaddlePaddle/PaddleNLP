# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import paddle


def print_tensor_info(t, name):
    if t is not None:
        print(f"-- [print_tensor_info] {name}: shape={t.shape}, dtype={t.dtype}")
    else:
        print(f"-- [print_tensor_info] {name}: tensor is {t}")


def unzip_and_dequant_wint2_5(zipped_weight, super_scale, weight_dtype=None, scale_compute_dtype=None):
    # zipped_weight: [num_experts, zipped_in_feature_size, out_feature_size]
    # super_scale: [num_experts, out_feature_size]

    if weight_dtype is None:
        weight_dtype = super_scale.dtype

    if scale_compute_dtype is None:
        scale_compute_dtype = super_scale.dtype
    elif super_scale.dtype != scale_compute_dtype:
        super_scale = super_scale.cast(scale_compute_dtype)

    # print_tensor_info(zipped_weight, "input_zipped_weight")
    # print_tensor_info(super_scale, "input_super_scale")
    # print("input_super_scale:", paddle.cast(super_scale, "float32"))

    zipped_group_size = 10
    assert zipped_weight.shape[1] % zipped_group_size == 0
    num_experts = zipped_weight.shape[0]
    num_groups = zipped_weight.shape[1] // zipped_group_size
    out_feature_size = zipped_weight.shape[2]
    zipped_weight = zipped_weight.reshape([num_experts, num_groups, zipped_group_size, 1, out_feature_size])
    # print_tensor_info(zipped_weight, "reshaped_zipped_weight")

    # scale
    super_scale = super_scale.reshape([num_experts, 1, 1, 1, out_feature_size])
    # print_tensor_info(super_scale, "reshaped_super_scale")
    # print("-- [reference-unzip_wint2.5] reshaped_super_scale[1, 0, 0, 0, 0:6]:", super_scale[1, 0, 0, 0, 0:6])
    local_scale = zipped_weight[:, :, -1:, :, :]
    local_scale = local_scale.cast("int32") & paddle.to_tensor(0x1FFF, dtype="int32")
    local_scale = local_scale.cast(scale_compute_dtype)
    # print("-- [reference-unzip_wint2.5] local_scale[1, 0, 0, 0, 0:6]:", local_scale[1, 0, 0, 0, 0:6])
    scale = super_scale * local_scale
    # print("-- [reference-unzip_wint2.5] scale[0, 0, 0, 0, 0:6]:", scale[0, 0, 0, 0, 0:6])

    # unzip weight
    shifts = paddle.to_tensor([13, 11, 9, 6, 4, 2, 0]).unsqueeze(-1).cast("int32")
    mask = paddle.to_tensor(2**3 - 1, dtype="int32")
    weight = (zipped_weight.cast("int32") >> shifts) & mask
    # print("-- [reference-unzip_wint2.5] shifted_weight[1, 0, 0, 0, 0:6]:", weight[1, 0, 0, 0, 0:6])
    weight = (weight - 4).cast(scale_compute_dtype) * scale
    # print("-- [reference-unzip_wint2.5] unzipped_weight[1, 0, 0, 0, 0:6]:", weight[1, 0, 0, 0, 0:6])
    weight = weight.cast(weight_dtype)

    # final reshape
    weight = weight.reshape([num_experts, num_groups, -1, out_feature_size])
    unzipped_group_size = 64
    assert weight.shape[2] > unzipped_group_size
    weight = weight[:, :, 0:unzipped_group_size, :]
    weight = weight.reshape([num_experts, -1, out_feature_size])
    return weight


def moe_group_gemm(permute_input, token_nums_per_expert, weight):
    """
    weight: [num_experts, hidden_size, inter_dim]
    """
    # 1. 创建输出张量
    output = paddle.zeros((permute_input.shape[0], weight.shape[2]), dtype="bfloat16")

    # 2. 计算前缀和，仅用于token分配
    token_nums_per_expert_np = token_nums_per_expert.numpy()
    token_nums_prefix_sum_np = np.zeros(len(token_nums_per_expert_np) + 1, dtype=np.int64)
    token_nums_prefix_sum_np[1:] = np.cumsum(token_nums_per_expert_np)

    # 3. 为每个专家计算
    for expert_idx in range(len(token_nums_per_expert_np)):
        # 获取当前专家的token范围
        start_idx = token_nums_prefix_sum_np[expert_idx]
        end_idx = token_nums_prefix_sum_np[expert_idx + 1]

        if start_idx == end_idx:  # 该专家没有分配token
            continue

        # 获取该专家需要处理的输入和权重
        expert_input = permute_input[start_idx:end_idx]
        expert_w0 = weight[expert_idx]
        expert_out = paddle.matmul(expert_input, expert_w0)

        # 将结果存入最终输出
        output[start_idx:end_idx] = expert_out

    return output
