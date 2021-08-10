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

import numpy as np
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle.nn import Layer, Embedding
from .. import PretrainedModel, register_base_model

__all__ = [
    'BlenderbotSmallModel', 'BlenderbotSmallPretrainedModel',
    'BlenderbotSmallEncoder', 'BlenderbotSmallDecoder',
    'BlenderbotSmallForConditionalGeneration'
]


# Copied from paddlenlp.transformers.bart.modeling.shift_tokens_right
def shift_tokens_right(input_ids: tensor, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = paddle.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    return shifted_input_ids



class BlenderbotSmallLearnedPositionalEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__(
            num_embeddings,
            embedding_dim,
        )

    def forward(self, input_ids_shape, past_key_values_length=0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = paddle.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype="int64")
        return super().forward(positions)


class BlenderbotSmallPretrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading
    and loading pretrained models.
    """
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "blenderbot_small-90M": {
            "vocab_size": 54944,
            "bos_token_id": 1,
            "pad_token_id": 0,
            "eos_token_id": 2,
            "decoder_start_token_id": 1,
            "d_model": 512,
            "num_encoder_layers": 8,
            "num_decoder_layers": 8,
            "encoder_attention_heads": 16,
            "decoder_attention_heads": 16,
            "decoder_ffn_dim": 2048,
            "encoder_ffn_dim": 2048,
            "dropout": 0.1,
            "activation_function": "gelu",
            "init_std": 0.02,
            "max_position_embeddings": 512,
            "attention_dropout": 0.0,
            "activation_dropout": 0.0,
            "scale_embedding": True,
            "normalize_before": False,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "blenderbot_small-90M":
                "https://paddlenlp.bj.bcebos.com/models/transformers/blenderbot_small/blenderbot_small-90M.pdparams",
        }
    }
    base_model_prefix = "blenderbot_small"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.init_std if hasattr(self, "init_std") else
                        self.blenderbot_small.config["init_std"],
                        shape=layer.weight.shape))


class BlenderbotSmallEncoder(BlenderbotSmallPretrainedModel):
    def __init__(self,
                 embed_tokens,
                 vocab_size,
                 pad_token_id=0,
                 d_model=512,
                 num_encoder_layers=6,
                 encoder_attention_heads=12,
                 encoder_ffn_dim=2048,
                 dropout=0.1,
                 activation_function='gelu',
                 attention_dropout=0.0,
                 activation_dropout=0.0,
                 max_position_embeddings=1024,
                 init_std=0.02,
                 scale_embedding=True,
                 normalize_before=False
                 ):
        super().__init__()
        self.init_std = init_std
        self.pad_token_id = pad_token_id
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(vocab_size, d_model, pad_token_id)
        self.encoder_embed_positions = BlenderbotSmallLearnedPositionalEmbedding(
            max_position_embeddings, d_model, pad_token_id)
        self.embed_scale = math.sqrt(d_model) if scale_embedding else 1.0
        self.encoder_dropout = nn.Dropout(dropout)
        self.encoder_layernorm_embedding = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=encoder_attention_heads,
            dim_feedforward=encoder_ffn_dim,
            dropout=dropout,
            activation=activation_function,
            attn_dropout=attention_dropout,
            act_dropout=activation_dropout,
            normalize_before=normalize_before
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.apply(self.init_weights)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
    ):
        if input_ids is None:
            raise ValueError("Input_ids cannot be None.")
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        inputs_embed_pos = self.encoder_embed_positions(input_ids.shape)
        hidden_states = inputs_embeds + inputs_embed_pos
        hidden_states = self.encoder_layernorm_embedding(hidden_states)
        encoder_input = self.encoder_dropout(hidden_states)

        if attention_mask is None:
            attention_mask = paddle.cast(
                input_ids == self.pad_token_id,
                dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e9
            attention_mask.stop_gradient = True

        encoder_output = self.encoder(encoder_input, src_mask=attention_mask)
        return encoder_output


class BlenderbotSmallDecoder(BlenderbotSmallPretrainedModel):
    def __init__(self,
                 embed_tokens,
                 vocab_size,
                 pad_token_id=1,
                 d_model=768,
                 num_decoder_layers=6,
                 decoder_attention_heads=12,
                 decoder_ffn_dim=3072,
                 dropout=0.1,
                 activation_function='gelu',
                 attention_dropout=0.1,
                 activation_dropout=0.1,
                 max_position_embeddings=1024,
                 init_std=0.02,
                 scale_embedding=True,
                 normalize_before=False):
        super().__init__()
        self.init_std = init_std
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(vocab_size, d_model, pad_token_id)

        self.decoder_embed_positions = BlenderbotSmallLearnedPositionalEmbedding(
            max_position_embeddings, d_model, pad_token_id)
        self.decoder_dropout = nn.Dropout(dropout)
        self.decoder_layernorm_embedding = nn.LayerNorm(d_model)
        self.embed_scale = math.sqrt(d_model) if scale_embedding else 1.0

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=decoder_attention_heads,
            dim_feedforward=decoder_ffn_dim,
            dropout=dropout,
            activation=activation_function,
            attn_dropout=attention_dropout,
            act_dropout=activation_dropout,
            normalize_before=normalize_before)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.apply(self.init_weights)

    def forward(self,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                encoder_output=None,
                memory_mask=None,
                cache=None):
        if decoder_attention_mask is None:
            decoder_length = paddle.shape(decoder_input_ids)[-1]
            decoder_attention_mask = paddle.tensor.triu(
                (paddle.full(
                    (decoder_length, decoder_length),
                    -np.inf,
                    dtype=paddle.get_default_dtype())),
                1)
        decoder_inputs_embeds = self.embed_tokens(decoder_input_ids) * self.embed_scale
        decoder_inputs_embed_pos = self.decoder_embed_positions(decoder_input_ids.shape)

        # Different from BLenderbot, BlenderbotSmall Apply layer norm on decoder_inputs_embeds
        decoder_inputs_embeds = self.decoder_layernorm_embedding(decoder_inputs_embeds)

        hidden_states = decoder_inputs_embeds + decoder_inputs_embed_pos
        decoder_input = self.decoder_dropout(hidden_states)

        decoder_output = self.decoder(
            tgt=decoder_input,
            memory=encoder_output,
            tgt_mask=decoder_attention_mask,
            memory_mask=memory_mask,
            cache=cache)
        return decoder_output


@register_base_model
class BlenderbotSmallModel(BlenderbotSmallPretrainedModel):
    def __init__(self,
                 vocab_size,
                 bos_token_id=1,
                 pad_token_id=0,
                 eos_token_id=2,
                 decoder_start_token_id=1,
                 d_model=512,
                 num_encoder_layers=8,
                 num_decoder_layers=8,
                 encoder_attention_heads=16,
                 decoder_attention_heads=16,
                 encoder_ffn_dim=2048,
                 decoder_ffn_dim=2048,
                 dropout=0.1,
                 activation_function='gelu',
                 attention_dropout=0.0,
                 activation_dropout=0.0,
                 max_position_embeddings=512,
                 init_std=0.02,
                 scale_embedding=True,
                 normalize_before=False
                 ):
        super().__init__()
        self.init_std = init_std
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.shared = nn.Embedding(vocab_size, d_model, pad_token_id)
        self.encoder = BlenderbotSmallEncoder(
            self.shared, vocab_size, pad_token_id, d_model, num_encoder_layers,
            encoder_attention_heads, encoder_ffn_dim, dropout,
            activation_function, attention_dropout, activation_dropout,
            max_position_embeddings, init_std, scale_embedding, normalize_before)

        self.decoder = BlenderbotSmallDecoder(
            self.shared, vocab_size, pad_token_id, d_model, num_decoder_layers,
            decoder_attention_heads, decoder_ffn_dim, dropout,
            activation_function, attention_dropout, activation_dropout,
            max_position_embeddings, init_std, scale_embedding, normalize_before)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                encoder_output=None,
                cache=None):
        # automatically creates decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(input_ids,
                                                   self.decoder_start_token_id)
        if encoder_output is None:
            encoder_output = self.encoder(input_ids, attention_mask)
        memory_mask = paddle.cast(
            input_ids == self.pad_token_id,
            dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e9
        memory_mask.stop_gradient = True

        decoder_output = self.decoder(decoder_input_ids, decoder_attention_mask,
                                      encoder_output, memory_mask, cache)
        return decoder_output


# Copied from .paddlenlp.transformers.bart.modeling.BartForConditionalGeneration
# with Bart -> BlenderbotSmall
class BlenderbotSmallForConditionalGeneration(BlenderbotSmallPretrainedModel):
    def __init__(self, blenderbot_small):
        super().__init__()
        self.blenderbot_small = blenderbot_small
        self.lm_head_weight = self.create_parameter(
            shape=[
                self.blenderbot_small.config['vocab_size'], self.blenderbot_small.config['d_model']
            ],
            dtype=self.blenderbot_small.shared.weight.dtype,
            is_bias=False)
        self.register_buffer("final_logits_bias",
                             paddle.zeros((1, self.blenderbot_small.config['vocab_size'])))
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                encoder_output=None,
                cache=None):
        output = self.blenderbot_small(input_ids, attention_mask, decoder_input_ids,
                                       decoder_attention_mask, encoder_output, cache)
        lm_logits = paddle.tensor.matmul(
            output if cache is None else output[0],
            self.lm_head_weight,
            transpose_y=True) + self.final_logits_bias

        return lm_logits
