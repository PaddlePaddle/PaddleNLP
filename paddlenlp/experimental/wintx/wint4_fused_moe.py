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
"""
WINT4 cutlass implementation, faster than triton version
"""
from paddlenlp_ops import moe_expert_dispatch, moe_expert_ffn, moe_expert_reduce

__all__ = [
    "wint4_moe_pp",
]


def wint4_moe_pp(
    hidden_states,
    w1,
    w2,
    scores,
    topk,
    w1_scale,
    w2_scale,
):
    """
    WINT4 cutlass implementation, faster than triton version
    """
    (
        permute_input,
        token_nums_per_expert,
        permute_indices_per_token,
        top_k_weights,
        top_k_indices,
    ) = moe_expert_dispatch(hidden_states, scores, topk, False, topk_only_mode=True)

    ffn_out = moe_expert_ffn(
        permute_input,
        token_nums_per_expert,
        w1,
        w2,
        None,
        w1_scale,
        w2_scale,
        "weight_only_int4",
    )

    fused_moe_out = moe_expert_reduce(
        ffn_out,
        top_k_weights,
        permute_indices_per_token,
        top_k_indices,
        None,
        norm_topk_prob=False,
        routed_scaling_factor=1.0,
    )

    return fused_moe_out
