# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
""" ERNIE model configuration"""
from __future__ import annotations

from typing import Dict

from ..configuration_utils import PretrainedConfig

__all__ = ["ERNIE_PRETRAINED_INIT_CONFIGURATION", "ErnieConfig", "ERNIE_PRETRAINED_RESOURCE_FILES_MAP"]

ERNIE_PRETRAINED_INIT_CONFIGURATION = {
    "ernie-1.0": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 513,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 2,
        "vocab_size": 18000,
        "pad_token_id": 0,
    },
    "ernie-1.0-base-zh": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 513,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 2,
        "vocab_size": 18000,
        "pad_token_id": 0,
    },
    "ernie-1.0-base-zh-cw": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "task_type_vocab_size": 3,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "ernie-1.0-large-zh-cw": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 3072,  # it is 3072 instead of 4096
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "type_vocab_size": 2,
        "vocab_size": 18000,
        "pad_token_id": 0,
    },
    "ernie-tiny": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 600,
        "num_attention_heads": 16,
        "num_hidden_layers": 3,
        "type_vocab_size": 2,
        "vocab_size": 50006,
        "pad_token_id": 0,
    },
    "ernie-2.0-base-zh": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 513,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 4,
        "vocab_size": 18000,
    },
    "ernie-2.0-large-zh": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "intermediate_size": 4096,  # special for large model
        "initializer_range": 0.02,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "type_vocab_size": 4,
        "vocab_size": 12800,
    },
    "ernie-2.0-base-en": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 4,
        "vocab_size": 30522,
        "pad_token_id": 0,
    },
    "ernie-2.0-base-en-finetuned-squad": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 4,
        "vocab_size": 30522,
        "pad_token_id": 0,
    },
    "ernie-2.0-large-en": {
        "attention_probs_dropout_prob": 0.1,
        "intermediate_size": 4096,  # special for ernie-2.0-large-en
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "type_vocab_size": 4,
        "vocab_size": 30522,
        "pad_token_id": 0,
    },
    "rocketqa-zh-dureader-query-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 513,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 2,
        "vocab_size": 18000,
        "pad_token_id": 0,
    },
    "rocketqa-zh-dureader-para-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 513,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 2,
        "vocab_size": 18000,
        "pad_token_id": 0,
    },
    "rocketqa-v1-marco-query-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 4,
        "vocab_size": 30522,
        "pad_token_id": 0,
    },
    "rocketqa-v1-marco-para-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 4,
        "vocab_size": 30522,
        "pad_token_id": 0,
    },
    "rocketqa-zh-dureader-cross-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 513,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 2,
        "vocab_size": 18000,
        "pad_token_id": 0,
    },
    "rocketqa-v1-marco-cross-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 4,
        "vocab_size": 30522,
        "pad_token_id": 0,
    },
    "ernie-3.0-xbase-zh": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "intermediate_size": 4096,  # special for large model
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 16,
        "num_hidden_layers": 20,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "ernie-3.0-base-zh": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "task_type_vocab_size": 3,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "ernie-3.0-medium-zh": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "ernie-3.0-mini-zh": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 384,
        "intermediate_size": 1536,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "ernie-3.0-micro-zh": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 384,
        "intermediate_size": 1536,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 4,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "ernie-3.0-nano-zh": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 312,
        "intermediate_size": 1248,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 4,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "rocketqa-base-cross-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "task_type_vocab_size": 3,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "rocketqa-medium-cross-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "rocketqa-mini-cross-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 384,
        "intermediate_size": 1536,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "rocketqa-micro-cross-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 384,
        "intermediate_size": 1536,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 4,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "rocketqa-nano-cross-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 312,
        "intermediate_size": 1248,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 4,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "rocketqa-zh-base-query-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "task_type_vocab_size": 3,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "rocketqa-zh-base-para-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "task_type_vocab_size": 3,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "rocketqa-zh-medium-query-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "rocketqa-zh-medium-para-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "rocketqa-zh-mini-query-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 384,
        "intermediate_size": 1536,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "rocketqa-zh-mini-para-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 384,
        "intermediate_size": 1536,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "rocketqa-zh-micro-query-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 384,
        "intermediate_size": 1536,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 4,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "rocketqa-zh-micro-para-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 384,
        "intermediate_size": 1536,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 4,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "rocketqa-zh-nano-query-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 312,
        "intermediate_size": 1248,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 4,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "rocketqa-zh-nano-para-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 312,
        "intermediate_size": 1248,
        "initializer_range": 0.02,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 4,
        "task_type_vocab_size": 16,
        "type_vocab_size": 4,
        "use_task_id": True,
        "vocab_size": 40000,
    },
    "rocketqav2-en-marco-cross-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 4,
        "vocab_size": 30522,
        "pad_token_id": 0,
    },
    "rocketqav2-en-marco-query-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 4,
        "vocab_size": 30522,
        "pad_token_id": 0,
    },
    "rocketqav2-en-marco-para-encoder": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 4,
        "vocab_size": 30522,
        "pad_token_id": 0,
    },
    "ernie-search-base-dual-encoder-marco-en": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 4,
        "vocab_size": 30522,
        "pad_token_id": 0,
    },
    "ernie-search-large-cross-encoder-marco-en": {
        "attention_probs_dropout_prob": 0.1,
        "intermediate_size": 4096,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "type_vocab_size": 4,
        "vocab_size": 30522,
        "pad_token_id": 0,
    },
}

ERNIE_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        # Deprecated, alias for ernie-1.0-base-zh
        "ernie-1.0": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie/ernie_v1_chn_base.pdparams",
        "ernie-1.0-base-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie/ernie_v1_chn_base.pdparams",
        "ernie-1.0-base-zh-cw": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie/ernie_1.0_base_zh_cw.pdparams",
        "ernie-1.0-large-zh-cw": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie/ernie_1.0_large_zh_cw.pdparams",
        "ernie-tiny": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_tiny/ernie_tiny.pdparams",
        "ernie-2.0-base-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_2.0/ernie_2.0_base_zh.pdparams",
        "ernie-2.0-large-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_2.0/ernie_2.0_large_zh.pdparams",
        "ernie-2.0-base-en": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_v2_base/ernie_v2_eng_base.pdparams",
        "ernie-2.0-base-en-finetuned-squad": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_v2_base/ernie_v2_eng_base_finetuned_squad.pdparams",
        "ernie-2.0-large-en": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_v2_large/ernie_v2_eng_large.pdparams",
        "rocketqa-zh-dureader-query-encoder": "https://bj.bcebos.com/paddlenlp/models/transformers/rocketqa/rocketqa_zh_dureader_query_encoder.pdparams",
        "rocketqa-zh-dureader-para-encoder": "https://bj.bcebos.com/paddlenlp/models/transformers/rocketqa/rocketqa_zh_dureader_para_encoder.pdparams",
        "rocketqa-v1-marco-query-encoder": "https://bj.bcebos.com/paddlenlp/models/transformers/rocketqa/rocketqa_v1_marco_query_encoder.pdparams",
        "rocketqa-v1-marco-para-encoder": "https://bj.bcebos.com/paddlenlp/models/transformers/rocketqa/rocketqa_v1_marco_para_encoder.pdparams",
        "rocketqa-zh-dureader-cross-encoder": "https://bj.bcebos.com/paddlenlp/models/transformers/rocketqa/rocketqa_zh_dureader_cross_encoder.pdparams",
        "rocketqa-v1-marco-cross-encoder": "https://bj.bcebos.com/paddlenlp/models/transformers/rocketqa/rocketqa_v1_marco_cross_encoder.pdparams",
        "ernie-3.0-base-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_base_zh.pdparams",
        "ernie-3.0-xbase-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_xbase_zh.pdparams",
        "ernie-3.0-medium-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_medium_zh.pdparams",
        "ernie-3.0-mini-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_mini_zh.pdparams",
        "ernie-3.0-micro-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_micro_zh.pdparams",
        "ernie-3.0-nano-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_nano_zh.pdparams",
        "rocketqa-zh-base-query-encoder": "https://paddlenlp.bj.bcebos.com/models/transformers/rocketqa/rocketqa-zh-base-query-encoder.pdparams",
        "rocketqa-zh-base-para-encoder": "https://paddlenlp.bj.bcebos.com/models/transformers/rocketqa/rocketqa-zh-base-para-encoder.pdparams",
        "rocketqa-zh-medium-query-encoder": "https://paddlenlp.bj.bcebos.com/models/transformers/rocketqa/rocketqa-zh-medium-query-encoder.pdparams",
        "rocketqa-zh-medium-para-encoder": "https://paddlenlp.bj.bcebos.com/models/transformers/rocketqa/rocketqa-zh-medium-para-encoder.pdparams",
        "rocketqa-zh-mini-query-encoder": "https://paddlenlp.bj.bcebos.com/models/transformers/rocketqa/rocketqa-zh-mini-query-encoder.pdparams",
        "rocketqa-zh-mini-para-encoder": "https://paddlenlp.bj.bcebos.com/models/transformers/rocketqa/rocketqa-zh-mini-para-encoder.pdparams",
        "rocketqa-zh-micro-query-encoder": "https://paddlenlp.bj.bcebos.com/models/transformers/rocketqa/rocketqa-zh-micro-query-encoder.pdparams",
        "rocketqa-zh-micro-para-encoder": "https://paddlenlp.bj.bcebos.com/models/transformers/rocketqa/rocketqa-zh-micro-para-encoder.pdparams",
        "rocketqa-zh-nano-query-encoder": "https://paddlenlp.bj.bcebos.com/models/transformers/rocketqa/rocketqa-zh-nano-query-encoder.pdparams",
        "rocketqa-zh-nano-para-encoder": "https://paddlenlp.bj.bcebos.com/models/transformers/rocketqa/rocketqa-zh-nano-para-encoder.pdparams",
        "rocketqa-base-cross-encoder": "https://paddlenlp.bj.bcebos.com/models/transformers/rocketqa/rocketqa-base-cross-encoder.pdparams",
        "rocketqa-medium-cross-encoder": "https://paddlenlp.bj.bcebos.com/models/transformers/rocketqa/rocketqa-medium-cross-encoder.pdparams",
        "rocketqa-mini-cross-encoder": "https://paddlenlp.bj.bcebos.com/models/transformers/rocketqa/rocketqa-mini-cross-encoder.pdparams",
        "rocketqa-micro-cross-encoder": "https://paddlenlp.bj.bcebos.com/models/transformers/rocketqa/rocketqa-micro-cross-encoder.pdparams",
        "rocketqa-nano-cross-encoder": "https://paddlenlp.bj.bcebos.com/models/transformers/rocketqa/rocketqa-nano-cross-encoder.pdparams",
        "rocketqav2-en-marco-cross-encoder": "https://paddlenlp.bj.bcebos.com/models/transformers/rocketqa/rocketqav2_en_marco_cross_encoder.pdparams",
        "rocketqav2-en-marco-query-encoder": "https://paddlenlp.bj.bcebos.com/models/transformers/rocketqa/rocketqav2_en_marco_query_encoder.pdparams",
        "rocketqav2-en-marco-para-encoder": "https://paddlenlp.bj.bcebos.com/models/transformers/rocketqa/rocketqav2_en_marco_para_encoder.pdparams",
        "ernie-search-base-dual-encoder-marco-en": "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_search/ernie_search_base_dual_encoder_marco_en.pdparams",
        "ernie-search-large-cross-encoder-marco-en": "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_search/ernie_search_large_cross_encoder_marco_en.pdparams",
    }
}


class ErnieConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ErnieModel`]. It is used to
    instantiate a ERNIE model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ERNIE
    ernie-3.0-medium-zh architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the ERNIE model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ErnieModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`ErnieModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
    Examples:
    ```python
    >>> from paddlenlp.transformers import ErnieModel, ErnieConfig
    >>> # Initializing a ERNIE ernie-3.0-medium-zhstyle configuration
    >>> configuration = ErnieConfig()
    >>> # Initializing a model from the  style configuration
    >>> model = ErnieModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "ernie"
    attribute_map: Dict[str, str] = {"num_classes": "num_labels", "dropout": "classifier_dropout"}
    pretrained_init_configuration = ERNIE_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        task_type_vocab_size: int = 3,
        type_vocab_size: int = 16,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        pool_act: str = "tanh",
        fuse: bool = False,
        layer_norm_eps=1e-12,
        use_cache=False,
        use_task_id=True,
        enable_recompute=False,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.nmax_position_embeddingsum_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.task_type_vocab_size = task_type_vocab_size
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pool_act = pool_act
        self.fuse = fuse

        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.use_task_id = use_task_id
        self.enable_recompute = enable_recompute
