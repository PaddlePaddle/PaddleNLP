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

import paddle


def bloom_postprocess_past_key_value(past_key_values):
    # (layer_num, bs, head_num/tensor_parallel_degree, prefixlen, head_dim)*2
    past_key_values = paddle.transpose(past_key_values, perm=[2, 0, 3, 1, 4]).split(2)
    # (layer_num, bs, head_num/tensor_parallel_degree, prefixlen, head_dim)
    num_hidden_layers, batch_size, num_attention_heads, num_prefix_tokens, head_hidden_size = past_key_values[0].shape
    # (layer_num, bs, prefixlen, head_num/tensor_parallel_degree, head_dim)
    keys, values = past_key_values[0].transpose([0, 1, 3, 2, 4]), past_key_values[1].transpose([0, 1, 3, 2, 4])
    # (layer_num, bs*head_num/tensor_parallel_degree, head_dim, prefixlen)
    keys = keys.reshape([num_hidden_layers, batch_size * num_attention_heads, head_hidden_size, num_prefix_tokens])
    # (layer_num, bs*head_num/tensor_parallel_degree, prefixlen, head_dim)
    values = values.reshape([num_hidden_layers, batch_size * num_attention_heads, num_prefix_tokens, head_hidden_size])

    return tuple(zip(keys, values))


def chatglm_postprocess_past_key_value(past_key_values):
    # (layer_num, prefixlen, bs, head_num/tensor_parallel_degree, head_dim)*2
    keys, values = paddle.transpose(past_key_values, perm=[2, 1, 0, 3, 4]).split(2)

    return tuple(zip(keys, values))


def llama_postprocess_past_key_value(past_key_values):
    # (layer_num, bs, prefixlen, head_num/tensor_parallel_degree, head_dim)*2
    keys, values = paddle.transpose(past_key_values, perm=[2, 0, 1, 3, 4]).split(2)

    return tuple(zip(keys, values))


def chatglm_pad_attention_mask(input_ids_shape, num_prefix_tokens, attention_mask):
    prefix_attention_mask = paddle.ones(
        [input_ids_shape[0], 1, input_ids_shape[-1], num_prefix_tokens], dtype=attention_mask.dtype
    )
    prefix_attention_mask = (prefix_attention_mask < 0.5).astype("int64")
    return paddle.concat((prefix_attention_mask, attention_mask), axis=3)
