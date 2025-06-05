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

import paddle


def print_tensor_info(t, name):
    if t is not None:
        print(f"-- [print_tensor_info] {name}: shape={t.shape}, dtype={t.dtype}")
    else:
        print(f"-- [print_tensor_info] {name}: tensor is {t}")


def w_round(x):
    return paddle.floor(x + 0.5)


def unzip_and_dequant_wint2(w, w_scale, w_code_scale, w_code_zp, w_super_scale=None):
    """
    w                 uint8             [num_experts, in_feature_size // pack_num, out_feature_size]
    w_scale                             [num_experts, in_feature_size // group_size, out_feature_size]
    w_code_scale      float32           [num_experts, out_feature_size]
    w_code_zp         float32           [num_experts, out_feature_size]
    w_super_scale     w_scale.dtype     [num_experts, out_feature_size]

    output:           w_scale.dtype     [num_experts, in_feature_size, out_feature_size]
    """
    # step0: w dtype: uint8, shape: [num_experts, in_feature_size // pack_num, out_feature_size]
    # where pack_num = 4
    pack_num = 4
    bzp = 32
    num_experts, pack_in_feature_size, out_feature_size = w.shape

    in_feature_size = pack_in_feature_size * pack_num
    # step1: w need to unzip to shape: [num_experts, in_feature_size, out_feature_size]
    # here we use broadcast operation to implcitly expand the last dimension

    w = w.transpose(perm=[0, 2, 1]).reshape([num_experts, out_feature_size, pack_in_feature_size, 1])

    # for support repeat_interleave, w cast to int32
    w = w.cast("int32")
    w = w.repeat_interleave(pack_num, axis=-1)
    w = w.reshape([num_experts, out_feature_size, in_feature_size])
    w = w.transpose(perm=[0, 2, 1])

    # step2: w need to first dequant
    # w_code_scale shape: [num_experts, out_feature_size]
    # w_code_zp shape: [num_experts, out_feature_size]
    w_code_scale = w_code_scale.reshape([num_experts, 1, out_feature_size])
    w_code_zp = w_code_zp.reshape([num_experts, 1, out_feature_size])

    w = w_round(w.cast("float32") * w_code_scale + w_code_zp).cast("int16")

    # step3: w need to shifted and mask the original weight to unzip
    bit_shift = paddle.to_tensor([9, 6, 3, 0], dtype="int16")
    in_feature_bit_shift = bit_shift[paddle.arange(in_feature_size) % pack_num]
    in_feature_bit_shift = in_feature_bit_shift.reshape([1, in_feature_size, 1])
    mask = paddle.to_tensor(0x3F, dtype="int16")

    # step4: w need to shift and mask and second dequant
    w = ((w >> in_feature_bit_shift) & mask).cast(w_scale.dtype)

    if w_super_scale is not None:

        # w_super_scale shape: [num_experts, out_feature_size]
        # w_scale shape: [num_experts, in_feature_size // group_size,out_feature_size]
        # group_size = 64
        w_super_scale = w_super_scale.reshape([num_experts, 1, out_feature_size])
        w_scale = w_scale * w_super_scale

    # w_scale reshape to [num_experts, in_feature_size, out_feature_size]
    group_size = 64
    w_scale = w_scale.reshape([num_experts, in_feature_size // group_size, 1, out_feature_size])
    w_scale = w_scale.repeat_interleave(group_size, axis=2).reshape([num_experts, in_feature_size, out_feature_size])

    w = (w - bzp).cast(w_scale.dtype) * w_scale

    return w
