# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import TransformerEncoder

from .. import PretrainedModel, register_base_model

__all__ = [
    "UnifiedTransformerPretrainedModel",
    'UnifiedTransformerModel',
    'UnifiedTransformerLMHeadModel',
]


class UnifiedTransformerPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained UnifiedTransformer models. It provides 
    UnifiedTransformer related `model_config_file`, `resource_files_names`, 
    `pretrained_resource_files_map`, `pretrained_init_configuration`, 
    `base_model_prefix` for downloading and loading pretrained models. 
    See `PretrainedModel` for more details.
    """
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "unified_transformer-12L-cn": {
            "vocab_size": 30004,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "normalize_before": True,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "unk_token_id": 0,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "mask_token_id": 30000,
        },
        "unified_transformer-12L-cn-luge": {
            "vocab_size": 30004,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "normalize_before": True,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "unk_token_id": 0,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "mask_token_id": 30000,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "unified_transformer-12L-cn":
            "https://paddlenlp.bj.bcebos.com/models/transformers/unified_transformer/unified_transformer-12L-cn.pdparams",
            "unified_transformer-12L-cn-luge":
            "https://paddlenlp.bj.bcebos.com/models/transformers/unified_transformer/unified_transformer-12L-cn-luge.pdparams",
        }
    }
    base_model_prefix = "unified_transformer"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.unified_transformer.config["initializer_range"],
                        shape=layer.weight.shape))


class UnifiedTransformerEmbeddings(nn.Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2):
        super(UnifiedTransformerEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids, position_ids):
        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + position_embeddings + token_type_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


@register_base_model
class UnifiedTransformerModel(UnifiedTransformerPretrainedModel):
    def __init__(
            self,
            vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            normalize_before=True,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            unk_token_id=0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            mask_token_id=30000, ):
        super(UnifiedTransformerModel, self).__init__()
        self.unk_token_id = unk_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id
        self.initializer_range = initializer_range

        self.embeddings = UnifiedTransformerEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0,
            normalize_before=normalize_before)
        encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers,
                                             encoder_norm)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids,
                position_ids,
                attention_mask,
                use_cache=False,
                cache=None):
        embedding_output = self.embeddings(input_ids, token_type_ids,
                                           position_ids)
        if use_cache:
            if cache is None:
                cache = self.encoder.gen_cache(embedding_output)
            sequence_output, cache = self.encoder(embedding_output,
                                                  attention_mask, cache)
            return sequence_output, cache
        else:
            sequence_output = self.encoder(embedding_output, attention_mask)
            return sequence_output


class UnifiedTransformerLMHead(nn.Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(UnifiedTransformerLMHead, self).__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.activation = getattr(nn.functional, activation)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder_weight = self.create_parameter(
            shape=[hidden_size, vocab_size],
            dtype=self.transform.weight.dtype,
            is_bias=True) if embedding_weights is None else embedding_weights
        self.decoder_bias = self.create_parameter(
            shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = paddle.reshape(hidden_states,
                                           [-1, hidden_states.shape[-1]])
            hidden_states = paddle.tensor.gather(hidden_states,
                                                 masked_positions)
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = paddle.tensor.matmul(
            hidden_states, self.decoder_weight,
            transpose_y=True) + self.decoder_bias
        return logits


class UnifiedTransformerLMHeadModel(UnifiedTransformerPretrainedModel):
    def __init__(self, unified_transformer):
        super(UnifiedTransformerLMHeadModel, self).__init__()
        self.unified_transformer = unified_transformer
        self.lm_head = UnifiedTransformerLMHead(
            self.unified_transformer.config["hidden_size"],
            self.unified_transformer.config["vocab_size"],
            self.unified_transformer.config["hidden_act"],
            self.unified_transformer.embeddings.word_embeddings.weight)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids,
                position_ids,
                attention_mask,
                masked_positions=None,
                use_cache=False,
                cache=None):
        outputs = self.unified_transformer(input_ids, token_type_ids,
                                           position_ids, attention_mask,
                                           use_cache, cache)
        sequence_output = outputs[0] if use_cache else outputs
        logits = self.lm_head(sequence_output, masked_positions)
        if use_cache:
            cache = outputs[1]
            return logits, cache
        else:
            return logits

    def adjust_logits_during_generation(self, logits):
        # pre-process distribution
        logits[:, self.unified_transformer.unk_token_id] = -1e9
        logits[:, self.unified_transformer.bos_token_id] = -1e9
        logits[:, self.unified_transformer.mask_token_id] = -1e9
        return logits

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      token_type_ids,
                                      position_ids,
                                      attention_mask,
                                      use_cache=False,
                                      cache=None,
                                      **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        if cache is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
            position_ids = position_ids[:, -1].unsqueeze(-1)
            attention_mask = attention_mask[:, :, -1, :].unsqueeze(2)

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "cache": cache
        }

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(getattr(self, self.base_model_prefix), name)
