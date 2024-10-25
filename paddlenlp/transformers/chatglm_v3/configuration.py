# encoding=utf-8
# Copyright (c) 2023 ChatGLM2-6B Model Team and PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.transformers import PretrainedConfig


class ChatGLMv3Config(PretrainedConfig):
    attribute_map = {
        "padded_vocab_size": "vocab_size",
        "num_layers": "num_hidden_layers",
        "ffn_hidden_size": "intermediate_size",
        "layernorm_epsilon": "rms_norm_eps",
        "seq_length": "max_seq_len",
                     }
    def __init__(
        self,
        vocab_size=65024,
        hidden_size=4096,
        intermediate_size=13696,
        max_position_embeddings=8192,
        seq_length=8192,
        num_hidden_layers=28,
        num_attention_heads=32,
        num_key_value_heads=2,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        use_cache=True,
        use_recompute=False,
        recompute_granularity="full",
        pp_recompute_interval=1,
        no_recompute_layers=None,
        fuse_attention_qkv=False,
        fuse_attention_ffn=False,
        use_flash_attention=False,
        use_fused_rms_norm=False,
        use_fused_rope=False,
        tensor_parallel_output=True,
        sequence_parallel=False,
        fuse_sequence_parallel_allreduce=False,
        virtual_pp_degree=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        alibi=False,
        rope_scaling_factor=1.0,
        rope_scaling_type=None,
        long_sequence_strategy_type=None,
        long_sequence_strategy_name=None,
        long_sequence_init_args=None,
        use_long_sequence_strategies=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.seq_length = seq_length
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta

        self.use_cache = use_cache
        self.use_recompute = use_recompute
        self.recompute_granularity = recompute_granularity
        self.no_recompute_layers = no_recompute_layers
        self.pp_recompute_interval = pp_recompute_interval
        self.fuse_attention_qkv = fuse_attention_qkv
        self.use_flash_attention = use_flash_attention
        self.fuse_attention_ffn = fuse_attention_ffn
        self.use_fused_rms_norm = use_fused_rms_norm
        self.tensor_parallel_output = tensor_parallel_output
        self.sequence_parallel = sequence_parallel
        self.fuse_sequence_parallel_allreduce = fuse_sequence_parallel_allreduce
        self.virtual_pp_degree = virtual_pp_degree

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.alibi = alibi

        self.use_fused_rope = use_fused_rope
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_scaling_type = rope_scaling_type

        self.long_sequence_strategy_type = long_sequence_strategy_type
        self.long_sequence_strategy_name = long_sequence_strategy_name
        self.long_sequence_init_args = {} if long_sequence_init_args is None else long_sequence_init_args
        self.use_long_sequence_strategies = use_long_sequence_strategies

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            tensor_parallel_output=tensor_parallel_output,
            **kwargs,
        )

    @property
    def rope(self):
        return not self.alibi
