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

from .wintx_fused_moe import fused_moe_wint4
from .wintx_fused_moe_decode import (
    fused_moe_wintx_decode_wint2_5,
    fused_moe_wintx_decode_wint2_75,
)
from .wintx_gemm import gemm_int4_paddle


def weight_only_linear_int4_symm(x, pack_w, bias, scale):
    H = x.shape[-1]
    other_shape = x.shape[:-1]
    x = x.reshape((-1, H))  # .contiguous()

    gemm = gemm_int4_paddle

    if bias is None:
        out = gemm(x, pack_w, scale)
    else:
        out = gemm(x, pack_w, scale) + bias
    return out.reshape(other_shape + out.shape[-1:])


def weight_only_linear_int4_decode_superbs_moe_symm(
    hidden_states,
    w1,
    w2,
    scores,
    topk: int,
    w1_scale=None,
    w2_scale=None,
):

    fused_moe = fused_moe_wintx_decode_wint2_75
    return fused_moe(hidden_states, w1, w2, scores, topk, w1_scale, w2_scale)


def weight_only_linear_int3_decode_superbs_moe_symm(
    hidden_states,
    w1,
    w2,
    scores,
    topk: int,
    w1_scale=None,
    w2_scale=None,
):
    fused_moe = fused_moe_wintx_decode_wint2_5
    return fused_moe(hidden_states, w1, w2, scores, topk, w1_scale, w2_scale)


def weight_only_linear_int4_moe_symm(
    hidden_states,
    w1,
    w2,
    scores,
    topk: int,
    w1_scale=None,
    w2_scale=None,
):
    fused_moe = fused_moe_wint4
    return fused_moe(hidden_states, w1, w2, scores, topk, w1_scale, w2_scale)
