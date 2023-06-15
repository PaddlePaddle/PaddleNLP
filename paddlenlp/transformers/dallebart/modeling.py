# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021-2022 The Fairseq Authors and The Google Flax
# Team Authors And The HuggingFace Inc. team and & DALL·E Mini team.
# All rights reserved.
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

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.common_ops_import import convert_dtype

from ...transformers import PretrainedModel, register_base_model
from ...utils.env import CONFIG_NAME
from ...utils.log import logger
from ..generation_utils import BeamSearchScorer
from .configuration import (
    DALLEBART_PRETRAINED_INIT_CONFIGURATION,
    DALLEBART_PRETRAINED_RESOURCE_FILES_MAP,
    DalleBartConfig,
)

__all__ = [
    "DalleBartModel",
    "DalleBartPretrainedModel",
    "DalleBartEncoder",
    "DalleBartDecoder",
    "DalleBartForConditionalGeneration",
]


def shift_tokens_right(input_ids, decoder_start_token_id):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = paddle.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    return shifted_input_ids


def _convert_attention_mask(attn_mask, dtype):
    """
    Convert the attention mask to the target dtype we expect.

    Parameters:
        attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.
        dtype (VarType): The target type of `attn_mask` we expect.

    Returns:
        Tensor: A Tensor with shape same as input `attn_mask`, with data type `dtype`.
    """
    if attn_mask is not None and attn_mask.dtype != dtype:
        attn_mask_dtype = convert_dtype(attn_mask.dtype)
        if attn_mask_dtype == "bool" or "int" in attn_mask_dtype:
            attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 1e4
        else:
            attn_mask = paddle.cast(attn_mask, dtype)
    return attn_mask


class DalleBartPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained Bart models. It provides DalleBart related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    base_model_prefix = "dallebart"
    model_config_file = CONFIG_NAME
    pretrained_init_configuration = DALLEBART_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = DALLEBART_PRETRAINED_RESOURCE_FILES_MAP
    config_class = DalleBartConfig

    def _init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding, DalleBartLearnedPositionalEmbedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.normal(
                        mean=0.0,
                        std=self.config.init_std,
                        shape=layer.weight.shape,
                    )
                )


class DalleBartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings, embedding_dim):
        # DalleBart is set up so that if padding_idx is specified then offset the embedding ids by 0
        # and adjust num_embeddings appropriately. Other models dont have this hack
        self.offset = 0
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape, past_key_values_length=0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        seq_len = input_ids_shape[1]
        positions = paddle.arange(past_key_values_length, past_key_values_length + seq_len, dtype="int64")
        # (gongenlei) For dygraph to static graph
        return nn.Embedding.forward(self, positions + self.offset)


class GLU(nn.Layer):
    """
    From "GLU Variants Improve Transformer" by https://arxiv.org/abs/2002.05202
    """

    def __init__(
        self,
        count_in_out: int,
        count_middle: int,
        activation_dropout: float,
        dropout: float,
        activation_function: str = "gelu",
        use_bias: bool = False,
    ):
        super().__init__()
        self.ln0 = nn.LayerNorm(count_in_out)
        self.ln1 = nn.LayerNorm(count_middle)
        self.fc0 = nn.Linear(count_in_out, count_middle, bias_attr=use_bias)
        self.fc1 = nn.Linear(count_in_out, count_middle, bias_attr=use_bias)
        self.fc2 = nn.Linear(count_middle, count_in_out, bias_attr=use_bias)
        self.dropout1 = nn.Dropout(activation_dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = getattr(F, activation_function)

    def forward(self, z):
        z = self.ln0(z)
        w = self.fc0(z)
        w = self.act(w)
        v = self.fc1(z)
        z = self.dropout1(self.ln1(w * v))
        z = self.dropout2(self.fc2(z))
        return z


class DalleBartEncoderLayer(nn.Layer):
    """
    The Encoder Layer of DalleBartEncoder. The arguments of DalleBartEncoderLayer can see :class:`DalleBartEncoder`.
    """

    def __init__(self, config: DalleBartConfig):
        super().__init__()
        assert config.d_model > 0, "Expected d_model to be greater than 0, " "but received {}".format(config.d_model)
        assert (
            config.encoder_attention_heads > 0
        ), "Expected encoder_attention_heads to be greater than 0, " "but received {}".format(
            config.encoder_attention_heads
        )
        assert config.encoder_ffn_dim > 0, "Expected encoder_ffn_dim to be greater than 0, " "but received {}".format(
            config.encoder_ffn_dim
        )

        attention_dropout = config.dropout if config.attention_dropout is None else config.attention_dropout
        activation_dropout = config.dropout if config.activation_dropout is None else config.activation_dropout
        self.self_attn = nn.MultiHeadAttention(
            config.d_model, config.encoder_attention_heads, dropout=attention_dropout, bias_attr=config.use_bias
        )
        self.glu = GLU(
            config.d_model,
            config.encoder_ffn_dim,
            activation_dropout,
            config.dropout,
            config.activation_function,
            use_bias=config.use_bias,
        )

        self.pre_self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.dropout1 = nn.Dropout(config.dropout)

    def forward(self, src, src_mask=None):
        src_mask = _convert_attention_mask(src_mask, src.dtype)
        residual = src

        # pre_self_attn_layer_norm
        src = self.pre_self_attn_layer_norm(src)
        src = self.self_attn(src, src, src, src_mask)

        # self_attn_layer_norm
        src = self.self_attn_layer_norm(src)
        src = residual + self.dropout1(src)

        residual = src
        src = self.glu(src)
        src = residual + src
        return src


class DalleBartEncoder(DalleBartPretrainedModel):
    """
    The Encoder of DalleBartModel. The arguments of DalleBartEncoder can see :class:`DalleBartModel`.
    """

    def __init__(self, config: DalleBartConfig):
        super().__init__(config)
        self.init_std = config.init_std
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_positions = DalleBartLearnedPositionalEmbedding(config.max_text_length, config.d_model)

        self.layers = nn.LayerList([DalleBartEncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.final_ln = nn.LayerNorm(config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.text_pad_token_id = config.text_pad_token_id

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        The DalleBartEncoder forward method, overrides the `__call__()` special method.
        Args:
            input_ids (Tensor, optional):
                See :class:`DalleBartModel`.
            attention_mask (Tensor, optional):
                See :class:`DalleBartModel`.
        Returns:
            Tensor: Returns tensor `encoder_output`, which is the output at the last layer of the model.
            Its data type should be float32 and has a shape of [batch_size, sequence_length, hidden_size].
        """
        if input_ids is None:
            raise ValueError("Input_ids cannot be None.")

        if attention_mask is None:
            attention_mask = (
                paddle.cast(input_ids == self.text_pad_token_id, dtype=paddle.get_default_dtype()).unsqueeze([1, 2])
                * -1e4
            )
        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4
        attention_mask.stop_gradient = True

        inputs_embeds = self.embed_tokens(input_ids)
        inputs_embed_pos = self.embed_positions(input_ids.shape)
        hidden_states = self.layernorm_embedding(inputs_embeds + inputs_embed_pos)
        hidden_states = self.embedding_dropout(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        hidden_states = self.final_ln(hidden_states)

        return hidden_states


class DalleBartDecoderLayer(nn.Layer):
    """
    The Decoder Layer of DalleBartDecoder. The arguments of DalleBartDecoderLayer can see :class:`DalleBartDecoder`.
    """

    def __init__(self, config: DalleBartConfig):
        super().__init__()

        assert config.d_model > 0, "Expected d_model to be greater than 0, " "but received {}".format(config.d_model)
        assert (
            config.decoder_attention_heads > 0
        ), "Expected decoder_attention_heads to be greater than 0, " "but received {}".format(
            config.decoder_attention_heads
        )
        assert config.decoder_ffn_dim > 0, "Expected decoder_ffn_dim to be greater than 0, " "but received {}".format(
            config.decoder_ffn_dim
        )

        attention_dropout = config.dropout if config.attention_dropout is None else config.attention_dropout
        activation_dropout = config.dropout if config.activation_dropout is None else config.activation_dropout

        self.self_attn = nn.MultiHeadAttention(
            config.d_model, config.decoder_attention_heads, dropout=attention_dropout, bias_attr=config.use_bias
        )
        self.cross_attn = nn.MultiHeadAttention(
            config.d_model, config.decoder_attention_heads, dropout=attention_dropout, bias_attr=config.use_bias
        )

        self.glu = GLU(
            config.d_model,
            config.decoder_ffn_dim,
            activation_dropout,
            config.dropout,
            config.activation_function,
            use_bias=config.use_bias,
        )

        self.pre_self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.pre_cross_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.cross_attn_layer_norm = nn.LayerNorm(config.d_model)

        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):

        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)

        # self attn
        residual = tgt
        tgt = self.pre_self_attn_layer_norm(tgt)

        if cache is None:
            tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, None)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask, cache[0])

        tgt = self.self_attn_layer_norm(tgt)
        tgt = residual + self.dropout1(tgt)

        # cross attn
        residual = tgt
        tgt = self.pre_cross_attn_layer_norm(tgt)

        if cache is None:
            tgt = self.cross_attn(tgt, memory, memory, memory_mask, None)
        else:
            tgt, static_cache = self.cross_attn(tgt, memory, memory, memory_mask, cache[1])
        tgt = self.cross_attn_layer_norm(tgt)
        tgt = residual + self.dropout2(tgt)

        # glu
        residual = tgt
        tgt = self.glu(tgt)
        tgt = residual + tgt
        return tgt if cache is None else (tgt, (incremental_cache, static_cache))

    def gen_cache(self, memory):
        incremental_cache = self.self_attn.gen_cache(memory, type=self.self_attn.Cache)
        static_cache = self.cross_attn.gen_cache(memory, memory, type=self.cross_attn.StaticCache)
        return incremental_cache, static_cache


class DalleBartDecoder(DalleBartPretrainedModel):
    """
    The Decoder of DalleBartModel. The arguments of DalleBartDecoder can see :class:`DalleBartModel`.
    """

    def __init__(self, config: DalleBartConfig):
        super().__init__(config)
        self.init_std = config.init_std
        self.embed_tokens = nn.Embedding(config.image_vocab_size + 1, config.d_model)

        self.embed_positions = DalleBartLearnedPositionalEmbedding(config.max_image_length, config.d_model)
        self.layers = nn.LayerList([DalleBartDecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.final_ln = nn.LayerNorm(config.d_model)

    def forward(
        self,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_output=None,
        memory_mask=None,
        cache=None,
    ):
        """
        The DalleBartDecoder forward method, overrides the `__call__()` special method.
        Args:
            decoder_input_ids (Tensor, optional):
                See :class:`DalleBartModel`.
            decoder_attention_mask (Tensor, optional):
                See :class:`DalleBartModel`.
            encoder_output (Tensor, optional):
                See :class:`DalleBartModel`.
            memory_mask (Tensor, optional):
                See :class:`DalleBartModel`.
            cache (Tensor, optional):
                See :class:`DalleBartModel`.
        Returns:
            Tensor: Returns tensor `decoder_output`, which is the output at the last layer of the model.
            Its data type should be float32 and has a shape of [batch_size, sequence_length, hidden_size].
        """
        if decoder_attention_mask is None:
            decoder_length = paddle.shape(decoder_input_ids)[-1]
            decoder_attention_mask = paddle.triu(
                (
                    paddle.full(
                        (decoder_length, decoder_length),
                        -1e4,
                        dtype=paddle.get_default_dtype(),
                    )
                ),
                1,
            )
        decoder_inputs_embeds = self.embed_tokens(decoder_input_ids)
        past_key_values_length = paddle.shape(cache[0][0].k)[2] if cache is not None else 0
        decoder_inputs_embed_pos = self.embed_positions(paddle.shape(decoder_input_ids), past_key_values_length)
        hidden_states = decoder_inputs_embeds + decoder_inputs_embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # layers
        new_caches = []
        for i, layer in enumerate(self.layers):
            if cache is None:
                hidden_states = layer(
                    hidden_states,
                    encoder_output,
                    tgt_mask=decoder_attention_mask,
                    memory_mask=memory_mask,
                    cache=None,
                )
            else:
                hidden_states, new_cache = layer(
                    hidden_states,
                    encoder_output,
                    tgt_mask=decoder_attention_mask,
                    memory_mask=memory_mask,
                    cache=cache[i],
                )
                new_caches.append(new_cache)

        hidden_states = self.final_ln(hidden_states)

        return hidden_states if cache is None else (hidden_states, new_caches)

    def gen_cache(self, memory, do_zip=False):
        cache = [layer.gen_cache(memory) for layer in self.layers]
        if do_zip:
            cache = list(zip(*cache))
        return cache


@register_base_model
class DalleBartModel(DalleBartPretrainedModel):
    def __init__(self, config: DalleBartConfig):
        super().__init__(config)
        self.init_std = config.init_std
        self.pad_token_id = config.pad_token_id
        self.decoder_start_token_id = config.decoder_start_token_id
        self.text_pad_token_id = 1  # encoder pad id must be 1
        self.encoder = DalleBartEncoder(config)

        self.decoder = DalleBartDecoder(config)

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.encoder.embed_tokens = value

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_output=None,
        use_cache=False,
        cache=None,
    ):
        r"""
        The DalleBartModel forward method, overrides the `__call__()` special method.
        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                For example, its shape can be  [batch_size, sequence_length], [batch_size, sequence_length, sequence_length],
                [batch_size, num_attention_heads, sequence_length, sequence_length].
                Defaults to `None`, which means nothing needed to be prevented attention to.
            decoder_input_ids (Tensor, optional):
                Indices of decoder input sequence tokens in the vocabulary.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to `None`, which means no `decoder_input_ids` is provided, the model will create the tensor
                by shifting the `input_ids` to the right.
            decoder_attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention to some unwanted positions in `decoder_input_ids`.
                Its data type and shape is the same as `attention_mask`. Defaults to `None`.
            encoder_output (tuple, optional):
                The output of the encoder, a tuple consists `last_hidden_state`, `hidden_states`(optional), `attentions`(optional).
                The data type of `last_hidden_state` is float32 and its shape is `[batch_size, sequence_length, hidden_size]`.
                `hidden_states` is hidden_states of all layers in the Transformer encoder. The length of `hidden_states` is `num_hidden_layers + 1`.
                For all element in the tuple, its data type should be float32 and its shape is [`batch_size, sequence_length, hidden_size`].
                `attentions` is attentions of all layers of in the Transformer encoder. The length of `attentions` is `num_hidden_layers`.
                For all element in the tuple, its data type should be float32 and its shape is [`batch_size, num_attention_heads, sequence_length, sequence_length`].
            use_cache (bool, optional):
                 Whether or not to use cache. Defaults to `False`. If set to `True`, key value states will be returned and
                 can be used to speed up decoding.
            cache (list, optional):
                It is a list, and each element in the list is a tuple `(incremental_cache, static_cache)`.
                See `TransformerDecoder.gen_cache <https://github.com/PaddlePaddle/Paddle/blob/release/2.1/python/paddle/nn/layer/transformer.py#L1060>`__ for more details.
                It is only used for inference and should be None for training.
                Default to `None`.
        Returns:
            Tensor: Returns tensor `decoder_output`, which is the output at the last layer of the model.
            Its data type should be float32 and has a shape of [batch_size, sequence_length, hidden_size].
        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import DalleBartModel, DalleBartTokenizer
                tokenizer = DalleBartTokenizer.from_pretrained('dalle-mini')
                model = DalleBartModel.from_pretrained('dalle-mini')
                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        """
        if input_ids is None and encoder_output is None:
            raise ValueError("You have to specify either input_ids or encoder_output")
        if decoder_input_ids is None:
            assert input_ids is not None, "input_ids should be " "specified when generating decoder_input_ids"
            decoder_input_ids = shift_tokens_right(input_ids, self.decoder_start_token_id)
        if attention_mask is None:
            assert input_ids is not None, "input_ids should be " "specified when generating attention_mask"
            attention_mask = (
                paddle.cast(input_ids == self.text_pad_token_id, dtype=paddle.get_default_dtype()).unsqueeze([1, 2])
                * -1e4
            )
        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4
            attention_mask.stop_gradient = True
        if encoder_output is None:
            encoder_output = self.encoder(input_ids, attention_mask)
        if use_cache:
            if cache is None:
                cache = self.decoder.gen_cache(encoder_output)
        else:
            cache = None
        decoder_output = self.decoder(
            decoder_input_ids,
            decoder_attention_mask,
            encoder_output,
            attention_mask,
            cache,
        )

        return decoder_output


class DalleBartForConditionalGeneration(DalleBartPretrainedModel):
    r"""
    DalleBart Model with a `language modeling` head on top.
    Args:
        config (:class:`DalleBartConfig`):
            An instance of DalleBartConfig used to construct DalleBartForConditionalGeneration.
    """

    def __init__(self, config: DalleBartConfig):
        super().__init__(config)
        self.dallebart = DalleBartModel(config)
        self.lm_head = nn.Linear(
            config.d_model,
            config.image_vocab_size + 1,
            bias_attr=config.use_bias,
        )
        # input_ids_uncond
        # [0, 2, 1, 1, 1,...,1]
        # attention_mask_uncond
        # [1, 1, 0, 0, 0,...,0]
        input_ids_uncond = [0, 2] + [1] * (config.max_text_length - 2)
        attention_mask_uncond = [1, 1] + [0] * (config.max_text_length - 2)
        if hasattr(self, "input_ids_uncond"):
            self.input_ids_uncond = paddle.to_tensor([input_ids_uncond], dtype="int64")
        else:
            self.register_buffer(
                "input_ids_uncond", paddle.to_tensor([input_ids_uncond], dtype="int64"), persistable=False
            )
        if hasattr(self, "attention_mask_uncond"):
            self.attention_mask_uncond = paddle.to_tensor([attention_mask_uncond], dtype="int64")
        else:
            self.register_buffer(
                "attention_mask_uncond", paddle.to_tensor([attention_mask_uncond], dtype="int64"), persistable=False
            )

    def get_encoder(self):
        return self.dallebart.get_encoder()

    def get_decoder(self):
        return self.dallebart.get_decoder()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_output=None,
        use_cache=False,
        cache=None,
    ):
        r"""
        The DalleBartForConditionalGeneration forward method, overrides the __call__() special method.
        Args:
            input_ids (Tensor):
                See :class:`DalleBartModel`.
            attention_mask (Tensor, optional):
                See :class:`DalleBartModel`.
            decoder_input_ids (Tensor, `optional`):
                See :class:`DalleBartModel`.
            decoder_attention_mask (Tensor, optional):
                See :class:`DalleBartModel`.
            encoder_output (Tensor, optonal):
                See :class:`DalleBartModel`.
            use_cache (bool, optional):
                See :class:`DalleBartModel`.
            cache (Tensor, optional):
                See :class:`DalleBartModel`.
        Returns:
            Tensor or tuple: Returns Tensor `lm_logits` if `use_cache` is `False`, otherwise, returns tuple (`lm_logits`, `cache`).
            With the fields:
            - `lm_logits` (Tensor):
                The generated sentence of the model.
                Its data type should be float32 and has a shape of [batch_size, sequence_length, vocab_size].
            - `cache` (Tensor):
                See :class:`DalleBartModel`.
        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import DalleBartForConditionalGeneration, DalleBartTokenizer
                tokenizer = DalleBartTokenizer.from_pretrained('dalle-mini')
                model = DalleBartForConditionalGeneration.from_pretrained('dalle-mini')
                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)
        """
        output = self.dallebart(
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_output,
            use_cache,
            cache,
        )
        lm_logits = self.lm_head(output)
        if use_cache:
            cache = output[1]
            return lm_logits, cache
        else:
            return lm_logits

    def prepare_decoder_input_ids_from_labels(self, labels):
        return shift_tokens_right(labels, self.config.decoder_start_token_id)

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        attention_mask=None,
        decoder_attention_mask=None,
        cache=None,
        use_cache=False,
        encoder_output=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if cache is not None:
            decoder_input_ids = decoder_input_ids[:, -1].unsqueeze(-1)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask[:, :, -1, :].unsqueeze(-2)

        return {
            "input_ids": None,
            "decoder_input_ids": decoder_input_ids,
            "encoder_output": encoder_output,
            "decoder_attention_mask": decoder_attention_mask,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "cache": cache,
        }

    def sample(
        self,
        input_ids,
        logits_processors,
        max_length,
        pad_token_id,
        eos_token_id,
        top_k=None,
        top_p=None,
        temperature=None,
        min_tokens_to_keep=1,
        condition_scale=1.0,
        model_kwargs_uncond=None,
        **model_kwargs
    ):
        def TopKProcess(probs, top_k, min_tokens_to_keep):
            top_k = min(max(top_k, min_tokens_to_keep), probs.shape[-1])
            # Remove all tokens with a probability less than the last token of the top-k
            topk_probs, _ = paddle.topk(probs, k=top_k)
            probs = paddle.where(probs >= topk_probs[:, -1:], probs, paddle.full_like(probs, 0.0))
            return probs

        def TopPProcess(probs, top_p, min_tokens_to_keep):
            sorted_probs = paddle.sort(probs, descending=True)
            sorted_indices = paddle.argsort(probs, descending=True)
            cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)

            # Remove tokens with cumulative probs above the top_p, But keep at
            # least min_tokens_to_keep tokens
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Set 'min_tokens_to_keep - 1' because the first token is kept
                sorted_indices_to_remove[:, : min_tokens_to_keep - 1] = 0
            # Keep the first token
            sorted_indices_to_remove = paddle.cast(sorted_indices_to_remove, dtype="int64")
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0

            # Scatter sorted tensors to original indexing
            sorted_indices = sorted_indices + paddle.arange(probs.shape[0]).unsqueeze(-1) * probs.shape[-1]
            condition = paddle.scatter(
                sorted_indices_to_remove.flatten(), sorted_indices.flatten(), sorted_indices_to_remove.flatten()
            )
            condition = paddle.cast(condition, "bool").reshape(probs.shape)
            probs = paddle.where(condition, paddle.full_like(probs, 0.0), probs)
            return probs

        batch_size, cur_len = input_ids.shape
        origin_len = cur_len
        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())

        while cur_len < max_length:
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # perform super conditioning
            # Source: @RiversHaveWings - https://twitter.com/RiversHaveWings/status/1478093658716966912?s=20&t=xdm-wZ61Wf7OLnE_NJHZ1w
            if condition_scale != 1.0:
                model_inputs_uncond = self.prepare_inputs_for_generation(input_ids, **model_kwargs_uncond)
                outputs_uncond = self(**model_inputs_uncond)
                logits_uncond = outputs_uncond[0] if isinstance(outputs_uncond, tuple) else outputs_uncond
                # [batch_size, vocab_size]
                logits_uncond = logits_uncond[:, -1, :]
                logits = logits_uncond + condition_scale * (logits - logits_uncond)

            else:
                outputs_uncond = None

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            logits = logits_processors(input_ids, logits)

            # sample
            origin_probs = F.softmax(logits)
            origin_probs = paddle.log(origin_probs)
            if temperature is not None and temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits)
            if top_k is not None and top_k != 0:
                probs = TopKProcess(probs, top_k, min_tokens_to_keep)
            if top_p is not None and top_p < 1.0:
                probs = TopPProcess(probs, top_p, min_tokens_to_keep)
            next_tokens = paddle.multinomial(probs)

            next_scores = paddle.index_sample(origin_probs, next_tokens)

            if eos_token_id is not None:
                next_tokens = paddle.where(unfinished_flag, next_tokens, paddle.full_like(next_tokens, pad_token_id))

            scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)

            cur_len += 1
            input_ids = paddle.concat([input_ids, next_tokens], axis=1)

            if eos_token_id is not None:
                unfinished_flag = paddle.logical_and(unfinished_flag, next_tokens != eos_token_id)

            # Stop when there is a </s> in all sentences
            if not paddle.any(unfinished_flag):
                break

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )

            if condition_scale != 1.0:
                model_kwargs_uncond = self.update_model_kwargs_for_generation(
                    outputs_uncond, model_kwargs_uncond, is_encoder_decoder=self.is_encoder_decoder
                )
            else:
                model_kwargs_uncond = None

        return input_ids[:, origin_len:], scores

    @paddle.no_grad()
    def generate(
        self,
        input_ids=None,
        max_length=256,
        min_length=256,
        decode_strategy="sampling",
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        num_beams=1,
        num_beam_groups=1,
        length_penalty=0.0,
        early_stopping=False,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        text_pad_token_id=1,
        decoder_start_token_id=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        num_return_sequences=1,
        diversity_rate=0.0,
        use_cache=True,
        use_fast=False,
        use_fp16_decoding=False,
        condition_scale=1.0,
        **model_kwargs
    ):
        r"""
        The interface for generation task. This method can generate sequences
        by using decoding strategy. Currently, there are three decoding
        strategies supported: "greedy_search", "sampling" and "beam_search".

        Args:
            input_ids (Tensor, optional): The input sequence ids for the
                generation. It is a Tensor with shape [batch_size, sequence_length].
                The data type should be int32 or int64. Default to None, which
                we will initialize it as a Tensor with shape [1, 1], filled
                with the value `bos_token_id`.
            max_length (int, optional): The maximum length of the sequence to
                be generated. Default to 256.
            min_length (int, optional): The minimum length of the sequence to
                be generated. Default to 256.
            decode_strategy (str, optional): The decoding strategy in generation.
                Currently, there are three decoding strategies supported:
                "greedy_search", "sampling" and "beam_search". Default to
                "sampling".
            temperature (float, optional): The value used to module the next
                token probabilities in the "sampling" strategy. Default to 1.0,
                which means no effect.
            top_k (int, optional): The number of highest probability tokens to
                keep for top-k-filtering in the "sampling" strategy. Default to
                0, which means no effect.
            top_p (float, optional): The cumulative probability for
                top-p-filtering in the "sampling" strategy. The value should
                satisfy :math:`0 <= top\_p < 1`. Default to 1.0, which means no
                effect.
            repetition_penalty (float, optional):
                The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details. Defaults to 1.0.
            num_beams (int, optional): The number of beams in the "beam_search"
                strategy. Default to 1.
            num_beam_groups (int, optional):
                Number of groups to divide `num_beams` into in order to use DIVERSE
                BEAM SEARCH. See `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__
                for more details. Default to 1.
            length_penalty (float, optional): The exponential penalty to the
                sequence length in the "beam_search" strategy. The larger this
                param is, the more that the model would generate shorter
                sequences. Default to 0.0, which means no penalty.
            early_stopping (bool, optional): Whether to stop searching in the
                "beam_search" strategy when at least `num_beams` sentences are
                finished per batch or not. Default to False.
            bos_token_id (int, optional): The id of the `bos_token`. Default to
                None.
            eos_token_id (int, optional): The id of the `eos_token`. Default to
                None.
            pad_token_id (int, optional): The id of the `pad_token`. Default to
                None.
            decoder_start_token_id (int, optional): The start token id for
                encoder-decoder models. Default to None.
            forced_bos_token_id (int, optional): The id of the token to force as
                the first generated token. Usually use for multilingual models.
                Default to None.
            forced_eos_token_id (int, optional): The id of the token to force as
                the last generated token. Default to None.
            num_return_sequences (int, optional): The number of returned
                sequences for each sequence in the batch. Default to 1.
            diversity_rate (float, optional): If num_beam_groups is 1, this is the
                diversity_rate for Diverse Siblings Search. See
                `this paper https://arxiv.org/abs/1611.08562`__ for more details.
                If not, this is the diversity_rate for DIVERSE BEAM SEARCH.
            use_cache: (bool, optional): Whether to use the model cache to
                speed up decoding. Default to True.
            use_fast: (bool, optional): Whether to use fast entry of model
                for FastGeneration. Default to False.
            use_fp16_decoding: (bool, optional): Whether to use fp16 for decoding.
                Only works when fast entry is avalible. Default to False.
            condition_scale (float, optional): The scale of super conditioning. See
                `this twitter <https://twitter.com/RiversHaveWings/status/1478093658716966912>`__
                Default to 1.0.
            model_kwargs (dict): It can be used to specify additional kwargs
                passed to the model.

        Returns:
            tuple[Tensor]: It is a tuple contains two elements: ids and scores.
            Each element is a Tensor.

            With the fields:

            - ids (Tensor):
                The ids of the generated sequences. It is a Tensor with shape
                [batch_size * num_return_sequences, sequence_length]. The data
                type is same as the input `input_ids`.
            - scores (Tensor):
                The scores of the generated sequences. It is a Tensor with shape
                [batch_size * num_return_sequences, 1]. The data type is float32
                or float64, which is the same as the parameters in the model.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import (
                    DalleBartForConditionalGeneration,
                    DalleBartTokenizer
                )

                # Initialize the model and tokenizer
                model_name_or_path = 'dalle-mini'
                model = DalleBartForConditionalGeneration.from_pretrained(model_name_or_path)
                tokenizer = DalleBartTokenizer.from_pretrained(model_name_or_path)

                # Prepare the model inputs.
                prompts = "graphite sketch of Elon Musk"
                tokenized_inputs = tokenizer(
                    prompts,
                    return_tensors="pd",
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    max_length=64,
                )

                # Generate 4 sequences by using "sampling" strategy (top_k=64, condition_scale=10.0)
                image_token_ids, scores = model.generate(
                    input_ids=tokenized_inputs['input_ids'],
                    attention_mask=tokenized_inputs['attention_mask'],
                    decode_strategy="sampling",
                    condition_scale=10.0,
                    top_k=64,
                    num_return_sequences=4)
                print(image_token_ids.shape, scores.shape)
                # [4, 256] [4, 1]
        """
        assert decode_strategy in [
            "greedy_search",
            "sampling",
            "beam_search",
        ], "`decode_strategy` must be one of 'greedy_search', 'sampling' or 'beam_search' but received {}.".format(
            decode_strategy
        )

        bos_token_id = bos_token_id if bos_token_id is not None else getattr(self, "bos_token_id", None)
        eos_token_id = eos_token_id if eos_token_id is not None else getattr(self, "eos_token_id", None)
        pad_token_id = pad_token_id if pad_token_id is not None else getattr(self, "pad_token_id", None)
        forced_bos_token_id = (
            forced_bos_token_id if forced_bos_token_id is not None else getattr(self, "forced_bos_token_id", None)
        )
        forced_eos_token_id = (
            forced_eos_token_id if forced_eos_token_id is not None else getattr(self, "forced_eos_token_id", None)
        )
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else getattr(self, "decoder_start_token_id", None)
        )

        if getattr(self, "_fast_entry", None) is not False and use_fast:
            args = locals()
            args.pop("self")
            args.pop("__class__", None)
            model_kwargs = args.pop("model_kwargs")
            args.update(model_kwargs)
            try:
                if not hasattr(self, "_fast_entry"):
                    self._build_fast(args)
                if self._fast_entry:
                    output = self._fast_entry(**args)
                    if isinstance(output, tuple):
                        output_ids, dummy_srore = output
                    else:
                        output_ids = output
                        # make result and faster result oneconsistent
                        dummy_srore = None
                    if decode_strategy == "beam_search":
                        output_ids = output_ids.transpose([1, 2, 0])
                        output_ids = output_ids[:, :num_return_sequences, :].reshape([-1, output_ids.shape[-1]])
                        if dummy_srore is not None:
                            dummy_srore = dummy_srore[:, :num_return_sequences].flatten()
                    else:
                        output_ids = output_ids.transpose([1, 0])
                    return output_ids, dummy_srore

            except Exception as e:
                args["model_kwargs"] = model_kwargs
                # Prevent self._convert_to_fast to throw Exception
                self._convert_to_fast(args)
                logger.warning(e)
                logger.warning("FastGeneration is not available, " "and the original version would be used instead.")

        # params check
        if input_ids is None:
            # Init `input_ids` with bos_token_id
            input_ids = self.prepare_input_ids_for_generation(bos_token_id)

        if model_kwargs.get("attention_mask", None) is None:
            # Init `attention_mask` depending on `text_pad_token_id`
            model_kwargs["attention_mask"] = self.prepare_attention_mask_for_generation(
                input_ids, text_pad_token_id, eos_token_id
            )

        self.is_encoder_decoder = hasattr(self, "encoder") and hasattr(self, "decoder")
        if self.is_encoder_decoder:

            if condition_scale != 1.0:
                assert decode_strategy == "sampling", "`do_sample` has to be True for super conditioning."
                assert num_beams == 1, "`num_beams` has to be 1 for super conditioning."
                input_ids_uncond = self.input_ids_uncond.expand_as(input_ids)
                model_kwargs_uncond = {"attention_mask": self.attention_mask_uncond.expand_as(input_ids)}
                model_kwargs_uncond = self.prepare_encoder_decoder_kwargs_for_generation(
                    input_ids_uncond,
                    model_kwargs_uncond,
                )
                model_kwargs_uncond["use_cache"] = use_cache
            else:
                model_kwargs_uncond = None

            model_kwargs = self.prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)

            # set input_ids as decoder_input_ids
            if "decoder_input_ids" in model_kwargs:
                input_ids = model_kwargs.pop("decoder_input_ids")
            else:
                input_ids = self.prepare_decoder_input_ids_for_generation(
                    input_ids, decoder_start_token_id, bos_token_id
                )

        if pad_token_id is None and eos_token_id is not None:
            print("Setting `pad_token_id` to `eos_token_id`:{} for " "open-end generation.".format(eos_token_id))
            pad_token_id = eos_token_id

        model_kwargs["use_cache"] = use_cache
        max_length += input_ids.shape[-1]
        min_length += input_ids.shape[-1]

        logits_processors = self.get_logits_processor(
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_rate=diversity_rate,
            repetition_penalty=repetition_penalty,
        )

        if decode_strategy == "greedy_search":
            if num_return_sequences > 1:
                raise ValueError(
                    "`num_return_sequences` has to be 1, but is {} "
                    "when doing greedy search.".format(num_return_sequences)
                )

            return self.greedy_search(
                input_ids, logits_processors, max_length, pad_token_id, eos_token_id, **model_kwargs
            )

        elif decode_strategy == "sampling":

            if num_return_sequences > 1:
                tmpinput_ids = input_ids.clone()
                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=num_return_sequences, **model_kwargs
                )

                if condition_scale != 1.0:
                    _, model_kwargs_uncond = self.expand_inputs_for_generation(
                        tmpinput_ids, expand_size=num_return_sequences, **model_kwargs_uncond
                    )

            return self.sample(
                input_ids,
                logits_processors,
                max_length,
                pad_token_id,
                eos_token_id,
                top_k,
                top_p,
                temperature,
                condition_scale=condition_scale,
                model_kwargs_uncond=model_kwargs_uncond,
                **model_kwargs,
            )

        elif decode_strategy == "beam_search":
            batch_size = input_ids.shape[0]
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to "
                    "`num_beams`. But received `num_return_sequences` is {}, "
                    "`num_beams` is {}".format(num_return_sequences, num_beams)
                )
            if num_beams <= 1:
                raise ValueError(
                    "`num_beams` has to be bigger than 1. But received "
                    "`num_beams` is {}. If `num_beams` is 1, `decode_strategy` "
                    "should be 'greedy_search'".format(num_beams)
                )
            if num_beam_groups > 1:
                diverse_beam_scorer = BeamSearchScorer(
                    batch_size=batch_size,
                    max_length=max_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    do_early_stopping=early_stopping,
                    num_beam_hyps_to_keep=num_return_sequences,
                    num_beam_groups=num_beam_groups,
                )

                # interleave with `num_beams`
                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=num_beams, **model_kwargs
                )

                return self.group_beam_search(
                    input_ids,
                    diverse_beam_scorer,
                    logits_processors,
                    max_length,
                    diversity_rate,
                    pad_token_id,
                    eos_token_id,
                    **model_kwargs,
                )
            else:
                beam_scorer = BeamSearchScorer(
                    batch_size=batch_size,
                    max_length=max_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    do_early_stopping=early_stopping,
                    num_beam_hyps_to_keep=num_return_sequences,
                )

                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=num_beams, **model_kwargs
                )

                return self.beam_search(
                    input_ids,
                    beam_scorer,
                    logits_processors,
                    max_length,
                    diversity_rate,
                    pad_token_id,
                    eos_token_id,
                    **model_kwargs,
                )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(getattr(self, self.base_model_prefix), name)


class ResnetBlock(nn.Layer):
    def __init__(self, log2_count_in: int, log2_count_out: int):
        super().__init__()
        m, n = 2**log2_count_in, 2**log2_count_out
        self.is_middle = m == n
        self.norm1 = nn.GroupNorm(2**5, m)
        self.conv1 = nn.Conv2D(m, n, 3, padding=1)
        self.norm2 = nn.GroupNorm(2**5, n)
        self.conv2 = nn.Conv2D(n, n, 3, padding=1)
        if not self.is_middle:
            self.nin_shortcut = nn.Conv2D(m, n, 1)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.swish(h)
        h = self.conv2(h)
        if not self.is_middle:
            x = self.nin_shortcut(x)
        return x + h


class AttentionBlock(nn.Layer):
    def __init__(self):
        super().__init__()
        n = 2**9
        self.norm = nn.GroupNorm(2**5, n)
        self.q = nn.Conv2D(n, n, 1)
        self.k = nn.Conv2D(n, n, 1)
        self.v = nn.Conv2D(n, n, 1)
        self.proj_out = nn.Conv2D(n, n, 1)

    def forward(self, x):
        n, m = 2**9, x.shape[0]
        h = x
        h = self.norm(h)
        k = self.k(h)
        v = self.v(h)
        q = self.q(h)
        k = k.reshape(shape=[m, n, -1])
        v = v.reshape(shape=[m, n, -1])
        q = q.reshape(shape=[m, n, -1])
        q = q.transpose(perm=[0, 2, 1])
        w = paddle.bmm(q, k)
        w /= n**0.5
        w = F.softmax(w, axis=2)
        w = w.transpose(perm=[0, 2, 1])
        h = paddle.bmm(v, w)
        token_count = int(math.sqrt(h.shape[-1]))
        h = h.reshape(shape=[m, n, token_count, token_count])
        h = self.proj_out(h)
        return x + h


class MiddleLayer(nn.Layer):
    def __init__(self):
        super().__init__()
        self.block_1 = ResnetBlock(9, 9)
        self.attn_1 = AttentionBlock()
        self.block_2 = ResnetBlock(9, 9)

    def forward(self, h):
        h = self.block_1(h)
        h = self.attn_1(h)
        h = self.block_2(h)
        return h


class Upsample(nn.Layer):
    def __init__(self, log2_count):
        super().__init__()
        n = 2**log2_count
        self.upsample = nn.UpsamplingNearest2D(scale_factor=2)
        self.conv = nn.Conv2D(n, n, 3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class UpsampleBlock(nn.Layer):
    def __init__(self, log2_count_in: int, log2_count_out: int, has_attention: bool, has_upsample: bool):
        super().__init__()
        self.has_attention = has_attention
        self.has_upsample = has_upsample

        self.block = nn.LayerList(
            [
                ResnetBlock(log2_count_in, log2_count_out),
                ResnetBlock(log2_count_out, log2_count_out),
                ResnetBlock(log2_count_out, log2_count_out),
            ]
        )

        if has_attention:
            self.attn = nn.LayerList([AttentionBlock(), AttentionBlock(), AttentionBlock()])

        if has_upsample:
            self.upsample = Upsample(log2_count_out)

    def forward(self, h):
        for j in range(3):
            h = self.block[j](h)
            if self.has_attention:
                h = self.attn[j](h)
        if self.has_upsample:
            h = self.upsample(h)
        return h


class Decoder(nn.Layer):
    def __init__(self):
        super().__init__()

        self.conv_in = nn.Conv2D(2**8, 2**9, 3, padding=1)
        self.mid = MiddleLayer()

        self.up = nn.LayerList(
            [
                UpsampleBlock(7, 7, False, False),
                UpsampleBlock(8, 7, False, True),
                UpsampleBlock(8, 8, False, True),
                UpsampleBlock(9, 8, False, True),
                UpsampleBlock(9, 9, True, True),
            ]
        )

        self.norm_out = nn.GroupNorm(2**5, 2**7)
        self.conv_out = nn.Conv2D(2**7, 3, 3, padding=1)

    def forward(self, z):
        z = self.conv_in(z)
        z = self.mid(z)

        for i in reversed(range(5)):
            z = self.up[i](z)

        z = self.norm_out(z)
        z = F.swish(z)
        z = self.conv_out(z)
        return z
