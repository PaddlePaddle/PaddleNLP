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

"""Configuration class for Yuan2.0 model"""

from paddlenlp.transformers.configuration_utils import PretrainedConfig


class YuanConfig(PretrainedConfig):
    model_type = "yuan"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=135040,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=24,
        num_attention_heads=32,
        hidden_act="silu",
        model_max_length=8192,
        initializer_range=0.02,
        tensor_parallel_output=False,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=77185,
        bos_token_id=77185,
        eos_token_id=77185,
        num_key_value_heads=None,
        tie_word_embeddings=True,
        sequence_parallel=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.model_max_length = model_max_length
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.tensor_parallel_output = tensor_parallel_output
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.sequence_parallel = sequence_parallel
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tensor_parallel_output=tensor_parallel_output,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
