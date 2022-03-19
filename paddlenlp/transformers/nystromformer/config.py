# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import json


class NystromformerConfig:
    def __init__(
            self,
            vocab_size=30000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu_new",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=510,
            type_vocab_size=2,
            segment_means_seq_len=64,
            num_landmarks=64,
            conv_kernel_size=65,
            inv_coeff_init_option=False,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            num_labels=2,
            chunk_size_feed_forward=0,
            add_cross_attention=False,
            output_attentions=False,
            output_hidden_states=False,
            problem_type=None, ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.segment_means_seq_len = segment_means_seq_len
        self.num_landmarks = num_landmarks
        self.conv_kernel_size = conv_kernel_size
        self.inv_coeff_init_option = inv_coeff_init_option
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels

        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.add_cross_attention = add_cross_attention
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.problem_type = problem_type

    def load_config_json(self, config_json):
        with open(config_json) as f:
            config_dict = json.load(f)
            setattr(self, 'config_dict', config_dict)
            for key, value in config_dict.items():
                setattr(self, key, value)

    def __str__(self):
        return_str = 'NystromformerConfig(\n'
        for attr in dir(self):
            if attr[:
                    2] != '__' and attr != 'load_config_json' and attr != 'config_dict':
                return_str += '\t' + attr + ':  ' + str(getattr(self,
                                                                attr)) + '\n'
        return_str += ')\n'
        return return_str
