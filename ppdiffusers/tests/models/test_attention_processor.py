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

import unittest

import paddle

from ppdiffusers.models.attention_processor import Attention, AttnAddedKVProcessor


class AttnAddedKVProcessorTests(unittest.TestCase):
    def get_constructor_arguments(self, only_cross_attention: bool = False):
        query_dim = 10

        if only_cross_attention:
            cross_attention_dim = 12
        else:
            # when only cross attention is not set, the cross attention dim must be the same as the query dim
            cross_attention_dim = query_dim

        return {
            "query_dim": query_dim,
            "cross_attention_dim": cross_attention_dim,
            "heads": 2,
            "dim_head": 4,
            "added_kv_proj_dim": 6,
            "norm_num_groups": 1,
            "only_cross_attention": only_cross_attention,
            "processor": AttnAddedKVProcessor(),
        }

    def get_forward_arguments(self, query_dim, added_kv_proj_dim):
        batch_size = 2

        hidden_states = paddle.rand((batch_size, query_dim, 3, 2))
        encoder_hidden_states = paddle.rand((batch_size, 4, added_kv_proj_dim))
        attention_mask = None

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "attention_mask": attention_mask,
        }

    def test_only_cross_attention(self):
        # self and cross attention

        paddle.seed(0)

        constructor_args = self.get_constructor_arguments(only_cross_attention=False)
        attn = Attention(**constructor_args)

        self.assertTrue(attn.to_k is not None)
        self.assertTrue(attn.to_v is not None)

        forward_args = self.get_forward_arguments(
            query_dim=constructor_args["query_dim"], added_kv_proj_dim=constructor_args["added_kv_proj_dim"]
        )

        self_and_cross_attn_out = attn(**forward_args)

        # only self attention

        paddle.seed(0)

        constructor_args = self.get_constructor_arguments(only_cross_attention=True)
        attn = Attention(**constructor_args)

        self.assertTrue(attn.to_k is None)
        self.assertTrue(attn.to_v is None)

        forward_args = self.get_forward_arguments(
            query_dim=constructor_args["query_dim"], added_kv_proj_dim=constructor_args["added_kv_proj_dim"]
        )

        only_cross_attn_out = attn(**forward_args)

        self.assertTrue((only_cross_attn_out != self_and_cross_attn_out).all())
