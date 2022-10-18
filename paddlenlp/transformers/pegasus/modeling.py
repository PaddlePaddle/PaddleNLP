# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The Google Authors and The HuggingFace Inc. team.
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

from functools import partial
import math
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle.nn import Layer, Embedding

from .. import PretrainedModel, register_base_model

__all__ = [
    'PegasusModel', 'PegasusPretrainedModel', 'PegasusEncoder',
    'PegasusDecoder', 'PegasusForConditionalGeneration'
]


def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = paddle.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")

    shifted_input_ids = paddle.where(shifted_input_ids == -100, pad_token_id,
                                     shifted_input_ids)
    return shifted_input_ids


class PegasusPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained Pegasus models. It provides Pegasus related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """
    pretrained_init_configuration = {}
    pretrained_resource_files_map = {}
    base_model_prefix = "pegasus"

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
                        self.pegasus.config["init_std"],
                        shape=layer.weight.shape))
            if hasattr(layer, "bias"):
                layer.bias.set_value(paddle.zeros_like(layer.bias))
        elif isinstance(layer, PegasusSinusoidalPositionalEmbedding):
            pass


class PegasusSinusoidalPositionalEmbedding(Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
             for pos in range(n_pos)])
        out.stop_gradient = True
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = np.sin(position_enc[:, 0::2])
        out[:, sentinel:] = np.cos(position_enc[:, 1::2])
        return out

    @paddle.no_grad()
    def forward(self, input_ids_shape, past_key_values_length=0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = paddle.arange(past_key_values_length,
                                  past_key_values_length + seq_len,
                                  dtype="int64")
        # (gongenlei) For dygraph to static graph
        return Embedding.forward(self, positions)


class PegasusEncoder(PegasusPretrainedModel):
    """
    The Transformer Encoder of PegasusModel. The arguments of PegasusEncoder can see :class:`PegasusModel`.
    """

    def __init__(self,
                 embed_tokens,
                 vocab_size,
                 pad_token_id=1,
                 d_model=768,
                 num_encoder_layers=6,
                 encoder_attention_heads=12,
                 encoder_ffn_dim=3072,
                 dropout=0.1,
                 activation_function='relu',
                 attention_dropout=0.1,
                 activation_dropout=0.1,
                 max_position_embeddings=1024,
                 scale_embedding=True,
                 init_std=0.02):
        super().__init__()
        self.init_std = init_std
        self.pad_token_id = pad_token_id
        self.embed_scale = math.sqrt(d_model) if scale_embedding else 1.0
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(vocab_size, d_model)

        self.encoder_embed_positions = PegasusSinusoidalPositionalEmbedding(
            max_position_embeddings, d_model)

        self.encoder_dropout = nn.Dropout(dropout)
        self.encoder_layernorm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=encoder_attention_heads,
            dim_feedforward=encoder_ffn_dim,
            dropout=dropout,
            activation=activation_function,
            attn_dropout=attention_dropout,
            act_dropout=activation_dropout,
            normalize_before=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.apply(self.init_weights)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        The PegasusEncoder forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor, optional):
                See :class:`PegasusModel`.
            attention_mask (Tensor, optional):
                See :class:`PegasusModel`.

        Returns:
            Tensor: Returns tensor `encoder_output`, which is the output at the last layer of the model.
            Its data type should be float32 and has a shape of [batch_size, sequence_length, hidden_size].

        """
        if input_ids is None:
            raise ValueError("Input_ids cannot be None.")
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        inputs_embed_pos = self.encoder_embed_positions(paddle.shape(input_ids))
        hidden_states = inputs_embeds + inputs_embed_pos
        encoder_input = self.encoder_dropout(hidden_states)

        if attention_mask is None:
            attention_mask = paddle.cast(
                input_ids == self.pad_token_id,
                dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e4
        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(
                attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4
        attention_mask.stop_gradient = True

        encoder_output = self.encoder(encoder_input, src_mask=attention_mask)
        encoder_output = self.encoder_layernorm(encoder_output)
        return encoder_output


class PegasusDecoder(PegasusPretrainedModel):
    """
    The Transformer Decoder of PegasusModel. The arguments of PegasusDecoder can see :class:`PegasusModel`.
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
                 activation_function='relu',
                 attention_dropout=0.1,
                 activation_dropout=0.1,
                 max_position_embeddings=1024,
                 scale_embedding=True,
                 init_std=0.02):
        super().__init__()
        self.init_std = init_std
        self.embed_scale = math.sqrt(d_model) if scale_embedding else 1.0
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(vocab_size, d_model)

        self.decoder_embed_positions = PegasusSinusoidalPositionalEmbedding(
            max_position_embeddings, d_model)
        self.decoder_dropout = nn.Dropout(dropout)
        self.decoder_layernorm = nn.LayerNorm(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=decoder_attention_heads,
            dim_feedforward=decoder_ffn_dim,
            dropout=dropout,
            activation=activation_function,
            attn_dropout=attention_dropout,
            act_dropout=activation_dropout,
            normalize_before=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.apply(self.init_weights)

    def forward(self,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                encoder_output=None,
                memory_mask=None,
                cache=None):
        """
        The PegasusDecoder forward method, overrides the `__call__()` special method.

        Args:
            decoder_input_ids (Tensor, optional):
                See :class:`PegasusModel`.
            decoder_attention_mask (Tensor, optional):
                See :class:`PegasusModel`.
            encoder_output (Tensor, optional):
                See :class:`PegasusModel`.
            memory_mask (Tensor, optional):
                See :class:`PegasusModel`.
            cache (Tensor, optional):
                See :class:`PegasusModel`.

        Returns:
            Tensor: Returns tensor `decoder_output`, which is the output at the last layer of the model.
            Its data type should be float32 and has a shape of [batch_size, sequence_length, hidden_size].

        """
        if decoder_attention_mask is None:
            decoder_length = paddle.shape(decoder_input_ids)[-1]
            decoder_attention_mask = paddle.tensor.triu((paddle.full(
                (decoder_length, decoder_length),
                -np.inf,
                dtype=paddle.get_default_dtype())), 1)

        decoder_inputs_embeds = self.embed_tokens(
            decoder_input_ids) * self.embed_scale
        past_key_values_length = paddle.shape(
            cache[0][0].k)[2] if cache is not None else 0
        decoder_inputs_embed_pos = self.decoder_embed_positions(
            paddle.shape(decoder_input_ids), past_key_values_length)
        hidden_states = decoder_inputs_embeds + decoder_inputs_embed_pos
        decoder_input = self.decoder_dropout(hidden_states)

        decoder_output = self.decoder(tgt=decoder_input,
                                      memory=encoder_output,
                                      tgt_mask=decoder_attention_mask,
                                      memory_mask=memory_mask,
                                      cache=cache)
        if cache is not None:
            new_cache = decoder_output[1]
            decoder_output = decoder_output[0]
        else:
            new_cache = None
        decoder_output = self.decoder_layernorm(decoder_output)
        return decoder_output, new_cache


@register_base_model
class PegasusModel(PegasusPretrainedModel):
    r"""
    The bare Pegasus Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `PegasusModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `PegasusModel`.
        bos_token (int, optional):
            The beginning of sequence token that was used during pretraining. Can be
            used a sequence classifier token.
            Defaults to `0`.
        pad_token_id(int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.
        eos_token (int, optional):
            A special token representing the end of a sequence that was used during pretraining.
            Defaults to `1`.
        forced_eos_token_id (int, optional):
            The id of the token to force as the last generated token when max_length is reached. 
            Usually set to eos_token_id. Defaults to `1`.
        decoder_start_token_id (int, optional):
            If an encoder-decoder model starts decoding with a different token than bos, the id of that token.
            Defaults to `0`.
        d_model (int, optional):
            Dimensionality of the embedding layer, encoder layer and decoder layer. Defaults to `1024`.
        num_encoder_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `16`.
        num_decoder_layers (int, optional):
            Number of hidden layers in the Transformer decoder. Defaults to `16`.
        encoder_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to `16`.
        decoder_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer decoder.
            Defaults to `16`.
        encoder_ffn_dim (int, optional):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
            to ff layers are firstly projected from `d_model` to `encoder_ffn_dim`,
            and then projected back to `d_model`. Typically `encoder_ffn_dim` is larger than `d_model`.
            Defaults to `4096`.
        decoder_ffn_dim (int, optional):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
            to ff layers are firstly projected from `d_model` to `decoder_ffn_dim`,
            and then projected back to `d_model`. Typically `decoder_ffn_dim` is larger than `d_model`.
            Defaults to `4096`.
        dropout (float, optional):
            The dropout probability used in all fully connected layers (pre-process and post-process of MHA and FFN sub-layer)
            in the encoders and decoders. Defaults to `0.1`.
        activation_function (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions are supported.
            Defaults to `"relu"`.
        attention_dropout (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers and decoder layers to drop some attention target.
            Defaults to `0.1`.
        activation_dropout (float, optional):
            The dropout probability used after FFN activation in all encoder layers and decoder layers.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `1024`.
        scale_embedding (bool, optional):
            Scale embeddings by diving by sqrt(d_model). Defaults to `True`.
        init_std (float, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Default to `0.02`.

    """

    def __init__(self,
                 vocab_size,
                 bos_token_id=0,
                 pad_token_id=0,
                 eos_token_id=1,
                 forced_eos_token_id=1,
                 decoder_start_token_id=0,
                 d_model=1024,
                 num_encoder_layers=16,
                 num_decoder_layers=16,
                 encoder_attention_heads=16,
                 decoder_attention_heads=16,
                 encoder_ffn_dim=4096,
                 decoder_ffn_dim=4096,
                 dropout=0.1,
                 activation_function='relu',
                 attention_dropout=0.1,
                 activation_dropout=0.1,
                 max_position_embeddings=1024,
                 scale_embedding=True,
                 init_std=0.02):
        super().__init__()
        self.init_std = init_std
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.shared = nn.Embedding(vocab_size, d_model)
        self.encoder = PegasusEncoder(
            self.shared, vocab_size, pad_token_id, d_model, num_encoder_layers,
            encoder_attention_heads, encoder_ffn_dim, dropout,
            activation_function, attention_dropout, activation_dropout,
            max_position_embeddings, scale_embedding, init_std)

        self.decoder = PegasusDecoder(
            self.shared, vocab_size, pad_token_id, d_model, num_decoder_layers,
            decoder_attention_heads, decoder_ffn_dim, dropout,
            activation_function, attention_dropout, activation_dropout,
            max_position_embeddings, scale_embedding, init_std)
        self.apply(self.init_weights)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value

    def forward(self,
                input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                encoder_output=None,
                use_cache=False,
                cache=None):
        r'''
        The PegasusModel forward method, overrides the `__call__()` special method.

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
                from paddlenlp.transformers import PegasusModel, PegasusTokenizer

                tokenizer = PegasusTokenizer.from_pretrained('bart-base')
                model = PegasusModel.from_pretrained('bart-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        '''
        if input_ids is None and encoder_output is None:
            raise ValueError(
                "You have to specify either input_ids or encoder_output")
        if decoder_input_ids is None:
            assert input_ids is not None, "input_ids should be " \
                                          "specified when generating decoder_input_ids"
            decoder_input_ids = shift_tokens_right(input_ids, self.pad_token_id,
                                                   self.decoder_start_token_id)
        if attention_mask is None:
            assert input_ids is not None, "input_ids should be " \
                                          "specified when generating attention_mask"
            attention_mask = paddle.cast(
                input_ids == self.pad_token_id,
                dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e4
        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(
                attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4
            attention_mask.stop_gradient = True
        if encoder_output is None:
            encoder_output = self.encoder(input_ids, attention_mask)

        if use_cache:
            if cache is None:
                cache = self.decoder.decoder.gen_cache(encoder_output)
        else:
            cache = None
        decoder_output, new_cache = self.decoder(decoder_input_ids,
                                                 decoder_attention_mask,
                                                 encoder_output, attention_mask,
                                                 cache)
        return decoder_output, new_cache


class PegasusForConditionalGeneration(PegasusPretrainedModel):
    r"""
    Pegasus Model with a `language modeling` head on top.

    Args:
        Pegasus (:class:`PegasusModel`):
            An instance of PegasusModel.
    """

    def __init__(self, pegasus):
        super().__init__()
        self.pegasus = pegasus
        self.lm_head_weight = self.create_parameter(
            shape=[
                self.pegasus.config['vocab_size'],
                self.pegasus.config['d_model']
            ],
            dtype=self.pegasus.shared.weight.dtype,
            is_bias=False)
        self.register_buffer(
            "final_logits_bias",
            paddle.zeros((1, self.pegasus.config['vocab_size'])))

        self.apply(self.init_weights)

    def get_encoder(self):
        return self.pegasus.get_encoder()

    def get_decoder(self):
        return self.pegasus.get_decoder()

    def prepare_faster_entry(self, kwargs):
        from paddlenlp.ops import FasterPegasus
        decode_strategy = kwargs.get('decode_strategy')
        use_fp16_decoding = kwargs.get('use_fp16_decoding', False)
        decoding_lib = kwargs.get('decoding_lib', None)
        enable_faster_encoder = kwargs.get('enable_faster_encoder', True)
        if decode_strategy == 'sampling' and kwargs.get(
                'top_k') != 0 and kwargs.get('top_p') != 1:
            raise AttributeError(
                    "Only topk sampling or topp sampling are supported. " \
                    "Topk sampling and topp sampling cannot be both applied in the faster version.")
        if kwargs['repetition_penalty'] != 1.0:
            # not support for repetition_penalty yet in the faster version
            raise AttributeError(
                "'repetition_penalty != 1' is not supported yet in the faster version"
            )
        self._faster_entry = FasterPegasus(
            self,
            use_fp16_decoding=use_fp16_decoding,
            decoding_lib=decoding_lib,
            enable_faster_encoder=enable_faster_encoder).forward
        return self._faster_entry

    def forward(self,
                input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                encoder_output=None,
                use_cache=False,
                cache=None,
                labels=None):
        r"""
        The PegasusForConditionalGeneration forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`PegasusModel`.
            attention_mask (Tensor, optional):
                See :class:`PegasusModel`.
            decoder_input_ids (Tensor, `optional`):
                See :class:`PegasusModel`.
            decoder_attention_mask (Tensor, optional):
                See :class:`PegasusModel`.
            encoder_output (Tensor, optonal):
                See :class:`PegasusModel`.
            use_cache (bool, optional):
                See :class:`PegasusModel`.
            cache (Tensor, optional):
                See :class:`PegasusModel`.

        Returns:
            Tensor or tuple: Returns Tensor `lm_logits` if `use_cache` is `False`, otherwise, returns tuple (`lm_logits`, `cache`).

            With the fields:

            - `lm_logits` (Tensor):
                The generated sentence of the model.
                Its data type should be float32 and has a shape of [batch_size, sequence_length, vocab_size].

            - `cache` (Tensor):
                See :class:`PegasusModel`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import PegasusForConditionalGeneration, PegasusTokenizer

                tokenizer = PegasusTokenizer.from_pretrained('bart-base')
                model = PegasusForConditionalGeneration.from_pretrained('bart-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

        """
        output, new_cache = self.pegasus(input_ids, attention_mask,
                                         decoder_input_ids,
                                         decoder_attention_mask, encoder_output,
                                         use_cache, cache)
        lm_logits = paddle.tensor.matmul(
            output, self.lm_head_weight,
            transpose_y=True) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.reshape((-1, self.pegasus.config['vocab_size'])),
                labels.reshape((-1, )))

        return lm_logits, new_cache, masked_lm_loss

    def prepare_decoder_input_ids_from_labels(self, labels):
        return shift_tokens_right(labels, self.pegasus.pad_token_id,
                                  self.pegasus.config['decoder_start_token_id'])

    def prepare_inputs_for_generation(self,
                                      decoder_input_ids,
                                      attention_mask=None,
                                      decoder_attention_mask=None,
                                      cache=None,
                                      use_cache=False,
                                      encoder_output=None,
                                      **kwargs):
        # cut decoder_input_ids if past is used
        if cache is not None:
            decoder_input_ids = decoder_input_ids[:, -1].unsqueeze(-1)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask[:, :,
                                                                -1, :].unsqueeze(
                                                                    2)

        return {
            "input_ids": None,
            "decoder_input_ids": decoder_input_ids,
            "encoder_output": encoder_output,
            "decoder_attention_mask": decoder_attention_mask,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "cache": cache
        }

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            try:
                return getattr(getattr(self, self.base_model_prefix), name)
            except AttributeError:
                try:
                    return getattr(self, self.base_model_prefix).config[name]
                except KeyError:
                    raise e
