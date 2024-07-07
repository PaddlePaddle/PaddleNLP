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

import paddle

CONST_INPUT_HOOK = "register_forward_pre_hook"
CONST_OUTPUT_HOOK = "register_forward_post_hook"  # register_forward_hook
CONST_GRAD_HOOK = "register_hook"


split_and_select = lambda x, num_slice, selct_index: paddle.split(x, num_slice, axis=-1)[selct_index]


def split_heads(tensor, num_heads, attn_head_size):
    """Splits hidden_size dim into attn_head_size and num_heads."""
    new_shape = tensor.shape[:-1] + [num_heads, attn_head_size]
    tensor = tensor.reshape(new_shape)
    return tensor.transpose([0, 2, 1, 3])  # (batch, head, seq_length, head_features)


split_half = lambda x, selct_index: paddle.split(x, 2, axis=-1)[selct_index]
split_three = lambda x, selct_index: paddle.split(x, 3, axis=-1)[selct_index]
split_head_and_permute = lambda x, num_head: split_heads(x, num_head, x.shape[-1] // num_head)
