# Copyright (c) 2023 Technology Innovation Institute (TII) and PaddlePaddle Authors. All Rights Reserved.
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

from ..configuration_utils import PretrainedConfig

RW_PRETRAINED_INIT_CONFIGURATION = {
    "model_state": {
        "tiiuae/falcon-7b": "https://bj.bcebos.com/paddlenlp/models/community/tiiuae/falcon-7b/model_state.pdparams",
        "tiiuae/falcon-7b-instruct": "https://bj.bcebos.com/paddlenlp/models/community/tiiuae/falcon-7b-instruct/model_state.pdparams",
        "OpenBuddy/openbuddy-falcon-7b-v5-fp16": "https://bj.bcebos.com/paddlenlp/models/community/OpenBuddy/openbuddy-falcon-7b-v5-fp16/model_state.pdparams",
    },
}


class RWConfig(PretrainedConfig):
    model_type = "RefinedWeb"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
    }

    pretrained_init_configuration = RW_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size=250880,
        hidden_size=64,
        n_layer=2,
        n_head=8,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=1,
        eos_token_id=2,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        multi_query=False,
        n_head_kv=None,
        bias=False,
        alibi=False,
        parallel_attn=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        # Backward compatibility with n_embed kwarg
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.multi_query = multi_query

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.n_head_kv = n_head if n_head_kv is None else n_head_kv
        self.alibi = alibi
        self.bias = bias
        self.parallel_attn = parallel_attn

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @property
    def head_dim(self):
        return self.hidden_size // self.n_head

    @property
    def rotary(self):
        return not self.alibi
