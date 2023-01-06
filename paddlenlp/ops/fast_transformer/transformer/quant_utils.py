# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


def weight_channel_wise_quantize(weight, max_range=127, use_fp16_decoding=False):
    w_shape = weight.shape
    w_scale = []
    ret_weight = np.zeros(shape=weight.shape, dtype="int32")

    for i in range(w_shape[1]):
        s = max(abs(weight[:, i]))
        w_scale.append(s)

        inv_s = 1.0 / s
        for j in range(w_shape[0]):
            ret_weight[j, i] = round(max_range * inv_s * weight[j, i])

    return ret_weight.astype("int8"), np.asarray(w_scale).astype("float16" if use_fp16_decoding else "float32")


def index_CUBLASLT_ORDER_COL4_4R2_8C(col_id, row_id, m_32):
    new_col = col_id >> 5
    right_half = 4 if ((col_id & 7) >= 4) else 0
    new_row = (
        ((((row_id >> 3) << 3) + ((row_id & 1) << 2) + ((col_id & 31) >> 3)) << 5)
        + ((right_half + ((row_id & 7) >> 1)) << 2)
        + (col_id & 3)
    )
    return new_col * m_32 + new_row


def index_CUBLASLT_ORDER_COL32_2R_4R4(col_id, row_id, m_32):
    new_col = col_id >> 5
    row_in_tile = row_id & 31
    col_in_tile = col_id & 31
    new_row = (
        ((row_id >> 5) << 10)
        + (((((((row_in_tile & 7) >> 1) << 2) + (row_in_tile >> 3)) << 1) + (row_in_tile & 1)) << 5)
        + col_in_tile
    )
    return new_col * m_32 + new_row


def weight_COL_channel_wise_quantize(weight, max_range=127, use_fp16_decoding=False, sm=75):
    w_shape = weight.shape
    w_scale = []
    ret_weight = np.zeros(shape=weight.shape, dtype="int32")
    tmp_weight = weight.copy().astype("float64")

    for i in range(w_shape[1]):
        s = max(abs(weight[:, i]))
        w_scale.append(s)

    for i in range(w_shape[0]):
        for j in range(w_shape[1]):
            scale = w_scale[j]
            ele = tmp_weight[i, j]
            if sm == 75:
                idx_in_COL = index_CUBLASLT_ORDER_COL4_4R2_8C(i, j, 32 * w_shape[1])
            else:
                idx_in_COL = index_CUBLASLT_ORDER_COL32_2R_4R4(i, j, 32 * w_shape[1])
            ret_weight[int(idx_in_COL / w_shape[1]), int(idx_in_COL % w_shape[1])] = round(ele * max_range / scale)

    return ret_weight.astype("int8"), np.asarray(w_scale).astype("float16" if use_fp16_decoding else "float32")
