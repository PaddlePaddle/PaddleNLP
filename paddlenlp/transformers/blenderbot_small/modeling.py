# encoding=utf-8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The Facebook, Inc. and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import math
import paddle
import paddle.nn as nn
import paddle.tensor as tensor
from paddle.nn import Embedding
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
    """
    This module learns positional embeddings up to a fixed maximum size.

    Please should refer to the superclass for more information regarding methods and arguments.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__(
            num_embeddings,
            embedding_dim, )

    def forward(self, input_ids_shape, past_key_values_length=0):
        """
        Generate positional embeddings up based on input_ids_shape.
        Args:
            input_ids_shape (`tuple`): expected to be [batch_size, sequence_length].
            past_key_values_length (`int`, optional): The length of past_key_value,
            which is used only when the ``use_cache=True`` during prediction generating.

        Returns:
            (Tensor): The generated positional embedding.
        """
        bsz, seq_len = input_ids_shape[:2]
        positions = paddle.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype="int64")
        return super().forward(positions)


class BlenderbotSmallPretrainedModel(PretrainedModel):
    r"""
    An abstract class for pretrained BlenderbotSmall models. It provides BlenderbotSmall related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models.
    Refer to :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
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
        if paddle.get_default_dtype() not in ['float32', 'float64']:
            # gaussian/standard_normal/randn/normal only supports [float32, float64]
            return
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
    """
    The encoder of BlenderbotSmall Model.
    Please refer to :class:`~paddlenlp.transformers.model_utils.PretrainedModel` or
    :class:`~paddlenlp.transformers.Blenderbot.BlenderbotSmallModel` for more information
    regarding methods and parameters.
    """

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
                 normalize_before=False):
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
            normalize_before=normalize_before)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.apply(self.init_weights)

    def forward(
            self,
            input_ids=None,
            attention_mask=None, ):
        """
        Returns:
            Tensor: The last hidden-states at the last layer of the encoder.
            It's data type should be `float` and has a shape of `(batch_size, seq_lens, hidden_size)`.
            ``seq_lens`` corresponds to the length of input sequence.
        """
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
    """
    The decoder of BlenderbotSmall Model.
    Please refer to :class:`~paddlenlp.transformers.model_utils.PretrainedModel` and
    :class:`~paddlenlp.transformers.Blenderbot.BlenderbotModel` for more information
    regarding methods and parameters.
    """

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
                use_cache=False,
                cache=None):
        """
        Please refer to :class:`~paddlenlp.transformers.Blenderbot.BlenderbotModel` for more
        information regarding the parameters.
        Returns:
            Tensor|tuple:
                If ``use_cache=False``, the return will be the last hidden state of decoder with shape
                of [batch_size, seq_lens, hidden_size]. ``seq_lens`` corresponds to the length of input sequence.
                Otherwise, the return will be a tuple of ``(decoder_output, cache)``. Please refer to
                class :class:`paddle.nn.TransformerDecoder` for more information regarding ``cache``.
        """
        if decoder_attention_mask is None:
            decoder_length = paddle.shape(decoder_input_ids)[-1]
            decoder_attention_mask = paddle.tensor.triu(
                (paddle.full(
                    (decoder_length, decoder_length),
                    -np.inf,
                    dtype=paddle.get_default_dtype())),
                1)
        decoder_inputs_embeds = self.embed_tokens(
            decoder_input_ids) * self.embed_scale
        decoder_inputs_embed_pos = self.decoder_embed_positions(
            decoder_input_ids.shape)

        # Different from BLenderbot, BlenderbotSmall Apply layer norm on decoder_inputs_embeds
        decoder_inputs_embeds = self.decoder_layernorm_embedding(
            decoder_inputs_embeds)

        hidden_states = decoder_inputs_embeds + decoder_inputs_embed_pos
        decoder_input = self.decoder_dropout(hidden_states)

        if use_cache:
            if cache is None:
                cache = self.decoder.gen_cache(memory=encoder_output)

        decoder_output = self.decoder(
            tgt=decoder_input,
            memory=encoder_output,
            tgt_mask=decoder_attention_mask,
            memory_mask=memory_mask,
            cache=cache)
        return decoder_output


@register_base_model
class BlenderbotSmallModel(BlenderbotSmallPretrainedModel):
    r"""
     Construct a bare Blenderbot Model.

     This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
     Check the superclass documentation for the generic methods and the library implements for all its model.

     This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
     /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
     and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
         vocab_size (`int`):
             Vocabulary size of the BlenderbotSmall model.
         bos_token_id (`int`, optional):
            The id for begging of sentences token. Defaults to ``1``.
         pad_token_id (`int`, optional):
            The id for padding token. Defaults to ``0``.
         eos_token_id (`int`, optional):
            The id for end of sentence token. Defaults to ``2``.
         decoder_start_token_id (`int`, optional):
            The id indicating the start of decoding sentence. Defaults to ``1``.
         d_model (`int`, optional):
            Dimensionality of the layers and the pooler layer. Defaults to ``512``.
         num_encoder_layers (`int`, optional):
            Number of Transformer encoder layers for BlenderbotSmallEncoder. Defaults to ``8``.
         num_decoder_layers (`int`, optional):
            Number of Transformer decoder layers for BlenderbotSmallDecoder. Defaults to ``8``.
         encoder_attention_heads (`int`, optional):
            Number of attention heads for each Transformer encoder layer in BlenderbotSmallEncoder.
            Defaults to ``16``.
         decoder_attention_heads (`int`, optional):
            Number of attention heads for each Transformer decoder layer in BlenderbotSmallDecoder.
            Defaults to ``16``.
         encoder_ffn_dim (`int`, optional):
            Dimensionality of the feed-forward layer for each Transformer encoder layer in
            BlenderbotSmallEncoder. Defaults to ``2048``.
         decoder_ffn_dim (`int`, optional):
            Dimensionality of the feed-forward layer for each Transformer dncoder layer in
            BlenderbotSmallDncoder. Defaults to ``2048``.
         dropout (`float`, optional):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
            Defaults to ``0.1``.
         activation_function (`str`, optional):
            The non-linear activation function (function or string) in the encoder and pooler.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to ``"gelu"``.
         attention_dropout (`float`, optional):
            The dropout ratio for the attention probabilities.
            Defaults to ``0.0``.
         activation_dropout (`float`, optional):
            The dropout ratio for activations inside the fully connected layer.
         max_position_embeddings (`int`, optional):,
            The max position index of an input sequence. Defaults to ``512``.
         init_std (`float`, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Defaults to ``0.02``.
         scale_embedding (`bool`, optional):
            Indicate whether to scale embeddings by diving by sqrt(d_model). Defaults to ``True``.
         normalize_before (bool, optional):
            Indicate whether to put layer normalization into preprocessing of MHA and FFN sub-layers.
            If True, pre-process is layer normalization and post-precess includes dropout,
            residual connection. Otherwise, no pre-process and post-precess includes dropout,
            residual connection, layer normalization. Defaults to ``False``.
    """

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
                 normalize_before=False):
        super().__init__()
        self.init_std = init_std
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.shared = nn.Embedding(vocab_size, d_model, pad_token_id)
        self.encoder = BlenderbotSmallEncoder(
            self.shared, vocab_size, pad_token_id, d_model, num_encoder_layers,
            encoder_attention_heads, encoder_ffn_dim, dropout,
            activation_function, attention_dropout, activation_dropout,
            max_position_embeddings, init_std, scale_embedding,
            normalize_before)

        self.decoder = BlenderbotSmallDecoder(
            self.shared, vocab_size, pad_token_id, d_model, num_decoder_layers,
            decoder_attention_heads, decoder_ffn_dim, dropout,
            activation_function, attention_dropout, activation_dropout,
            max_position_embeddings, init_std, scale_embedding,
            normalize_before)
        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                encoder_output=None,
                use_cache=False,
                cache=None,
                **kwargs):
        r"""
        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].

            attention_mask (Tensor, optional):
                Mask to indicate whether to perform attention on each input token or not.
                The values should be either 0 or 1. The attention scores will be set
                to **-infinity** for any positions in the mask that are **0**, and will be
                **unchanged** for positions that are **1**.

                - **1** for tokens that are **not masked**,
                - **0** for tokens that are **masked**.

                It's data type should be `float32` and has a shape of [batch_size, sequence_length].
                Defaults to `None`.

            decoder_input_ids (Tensor, optional):
                If not provided, ``decoder_input_ids`` will be automatically generated based
                on ``decoder_start_token_id`` and ``input_ids``.

            decoder_attention_mask (Tensor, optional):
                If not provided, the default ``decoder_attention_mask`` will be a tensor with
                upper triangular part being ``-np.inf``. the shape will be ``(decoder_length, decoder_length)``

            encoder_output (Tensor, optional):
                The output of encoder. If not provided, a new ``encoder_output`` will be generated
                from BlenderbotEncoder. Defaults to ``None``.

            use_cache (bool, optional):
                Indicates whether to use cache to speed up decoding. Defaults to ``False``

            cache (list, optional): It is a list, and each element in the list
                is a tuple( :code:`(incremental_cache, static_cache)` ). See
                `TransformerDecoder.gen_cache` for more details. It is only
                used for inference and should be None for training. Default None.
        Returns:
            tuple: A tuple of `decoder_output` and `encoder_output`.

        Example:
            .. code-block::

            import paddle
            from paddlenlp.transformers import BlenderbotSmallTokenizer, BlenderbotSmallModel

            # "blenderbot_small-90M" is pretrained weight of BlenderbotSmallForConditionalGeneration,
            # Therefore some weight of additional layers in BlenderbotSmallForConditionalGeneration
            # might not be loaded and used.
            pretrained_model_name = "blenderbot_small-90M"
            tokenizer = BlenderbotSmallTokenizer.from_pretrained(pretrained_model_name)
            model = BlenderbotSmallModel.from_pretrained(pretrained_model_name)

            sample_text = "My friends are cool but they eat too many carbs."
            inputs = tokenizer(sample_text, return_attention_mask=True, return_token_type_ids=False)
            inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
            decoder_output, encoder_output = model(**inputs)
        """
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
                                      encoder_output, memory_mask, use_cache,
                                      cache)
        # return encoder output for decoder to generate sequence.
        return decoder_output, encoder_output


class BlenderbotSmallForConditionalGeneration(BlenderbotSmallPretrainedModel):
    """
    Please refer to :class:`~paddlenlp.transformers.Blenderbot.BlenderbotModel` for more
    information regarding parameters.
    Return:
        Tensor|tuple: If ``use_cache=False``, the return will be a tensor with shape of
            [batch_size, seq_lens, hidden_size]. Otherwise, the return will be a tuple
            of ``(decoder_output, cache)``.
    Example:
        .. code-block::

            import paddle
            from paddlenlp.transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration

            pretrained_model_name = "blenderbot_small-90M"
            tokenizer = BlenderbotSmallTokenizer.from_pretrained(pretrained_model_name)
            model = BlenderbotSmallForConditionalGeneration.from_pretrained(pretrained_model_name)

            sample_text = "My friends are cool but they eat too many carbs."
            inputs = tokenizer(sample_text, return_attention_mask=True, return_token_type_ids=False)
            inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
            outputs = model(**inputs, use_cache=True)
            # outputs is a tuple of (lm_logits, cache) if ``use_cache=True``.
    """

    def __init__(self, blenderbot_small):
        super().__init__()
        self.eos_token_id = blenderbot_small.eos_token_id
        self.bos_token_id = blenderbot_small.bos_token_id
        self.pad_token_id = blenderbot_small.pad_token_id
        self.blenderbot_small = blenderbot_small
        self.lm_head_weight = self.create_parameter(
            shape=[
                self.blenderbot_small.config['vocab_size'],
                self.blenderbot_small.config['d_model']
            ],
            dtype=self.blenderbot_small.shared.weight.dtype,
            is_bias=False)
        self.register_buffer(
            "final_logits_bias",
            paddle.zeros(
                (1, self.blenderbot_small.config['vocab_size']),
                dtype=paddle.get_default_dtype()))
        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                encoder_output=None,
                use_cache=False,
                cache=None):
        decoder_outputs, encoder_output = self.blenderbot_small(
            input_ids, attention_mask, decoder_input_ids,
            decoder_attention_mask, encoder_output, use_cache, cache)

        lm_logits = paddle.tensor.matmul(
            decoder_outputs[0] if use_cache else decoder_outputs,
            self.lm_head_weight,
            transpose_y=True) + self.final_logits_bias
        if use_cache:
            cache = decoder_outputs[1]
            return lm_logits, cache
        return lm_logits

    def prepare_inputs_for_generation(self,
                                      decoder_input_ids,
                                      attention_mask=None,
                                      encoder_output=None,
                                      use_cache=True,
                                      cache=None,
                                      **kwargs):
        if cache is not None:
            decoder_input_ids = decoder_input_ids[:, -1:].unsqueeze(-1)

        return {
            "input_ids":
            None,  # during prediction, Encoder_output is provided, do not need input_ids.
            "decoder_input_ids": decoder_input_ids,
            "encoder_output": encoder_output,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "cache": cache
        }
