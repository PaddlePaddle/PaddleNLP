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
from paddlenlp_ops import moe_permute

E = 16  # Number of experts
M = 32
TOP_K = 4
block_m = 128
K = 1024
SEEDS = 0

# Set seed for reproducibility
paddle.seed(SEEDS)

a = paddle.randn((M, K), dtype=paddle.bfloat16) / 10
scores = paddle.randn((M, E), dtype=paddle.float32) / 10
topk_weights, topk_ids = paddle.topk(scores, k=TOP_K, axis=-1, sorted=False)
print(a)
print(topk_ids)
print(topk_weights)
token_expert_indices = paddle.arange(M * TOP_K, dtype="int64").reshape([TOP_K, M]).transpose([1, 0])
print(token_expert_indices)
result0, result1, result2, result3 = moe_permute(
    a, topk_weights, topk_ids.astype("int32"), token_expert_indices.astype("int32"), None, E, E, TOP_K, block_m
)
print(result0)
print(result1)
print(result2)
print(result3)

# sorted_a, sorted_token_ids, expert_ids, num_tokens_post_padded = preprocess_for_moe_v1(topk_ids, a, E, block_m, K)

# inv_perm = paddle.argsort(sorted_token_ids)
# print("sorted_token_ids: ", sorted_token_ids)
# print("sorted_a: ", sorted_a)
# sorted_token_ids = sorted_token_ids.clip(max=M * TOP_KS - 1)
# A = (
#         a.reshape([a.shape[0], -1, a.shape[1]])
#         .expand([-1, TOP_KS, -1])
#         .reshape([-1, a.shape[1]])
#     )
# print("A", A)
# print(A[sorted_token_ids])
