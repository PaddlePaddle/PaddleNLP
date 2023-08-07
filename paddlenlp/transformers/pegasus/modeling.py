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

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn as nn
from paddle import Tensor
from paddle.nn import Embedding, MultiHeadAttention

from ...utils.env import CONFIG_NAME
from .. import PretrainedModel, register_base_model
from ..model_outputs import ModelOutput
from .configuration import PEGASUS_PRETRAINED_INIT_CONFIGURATION, PegasusConfig

__all__ = [
    "PegasusModel",
    "PegasusPretrainedModel",
    "PegasusEncoder",
    "PegasusDecoder",
    "PegasusForConditionalGeneration",
]

PEGASUS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese",
    "IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese",
    "IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese-V1",
    "PaddlePaddle/Randeng-Pegasus-238M-Summary-Chinese-SSTIA",
    "PaddlePaddle/Randeng-Pegasus-523M-Summary-Chinese-SSTIA",
]

Cache = MultiHeadAttention.Cache
StaticCache = MultiHeadAttention.StaticCache


def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = paddle.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")

    shifted_input_ids = paddle.where(
        shifted_input_ids == -100, paddle.full_like(shifted_input_ids, pad_token_id), shifted_input_ids
    )
    return shifted_input_ids


class PegasusPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained Pegasus models. It provides Pegasus related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    model_config_file = CONFIG_NAME
    pretrained_init_configuration = PEGASUS_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = {}
    base_model_prefix = "pegasus"
    config_class = PegasusConfig

    def _init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.config.init_std,
                        shape=layer.weight.shape,
                    )
                )
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
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.stop_gradient = True
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = np.sin(position_enc[:, 0::2])
        out[:, sentinel:] = np.cos(position_enc[:, 1::2])
        return out

    @paddle.no_grad()
    def forward(self, input_ids_shape: Tuple, past_key_values_length: int = 0) -> Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = paddle.arange(past_key_values_length, past_key_values_length + seq_len, dtype="int64")
        # (gongenlei) For dygraph to static graph
        return Embedding.forward(self, positions)


class PegasusEncoder(PegasusPretrainedModel):
    """
    The Transformer Encoder of PegasusModel. The arguments of PegasusEncoder can see :class:`PegasusModel`.
    """

    def __init__(self, config: PegasusConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.init_std = config.init_std
        self.pad_token_id = config.pad_token_id
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        self.encoder_embed_positions = PegasusSinusoidalPositionalEmbedding(
            config.max_position_embeddings, config.d_model
        )

        self.encoder_dropout = nn.Dropout(config.dropout)
        self.encoder_layernorm = nn.LayerNorm(config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.encoder_attention_heads,
            dim_feedforward=config.encoder_ffn_dim,
            dropout=config.dropout,
            activation=config.activation_function,
            attn_dropout=config.attention_dropout,
            act_dropout=config.activation_dropout,
            normalize_before=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.encoder_layers)

    def forward(self, input_ids: Optional[Tensor] = None, attention_mask: Optional[Tensor] = None, **kwargs):
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
            attention_mask = (
                paddle.cast(input_ids == self.pad_token_id, dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e4
            )
        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4
        attention_mask.stop_gradient = True

        encoder_output = self.encoder(encoder_input, src_mask=attention_mask)
        encoder_output = self.encoder_layernorm(encoder_output)
        return encoder_output


class PegasusDecoder(PegasusPretrainedModel):
    """
    The Transformer Decoder of PegasusModel. The arguments of PegasusDecoder can see :class:`PegasusModel`.
    """

    def __init__(self, config: PegasusConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.init_std = config.init_std
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        self.decoder_embed_positions = PegasusSinusoidalPositionalEmbedding(
            config.max_position_embeddings, config.d_model
        )
        self.decoder_dropout = nn.Dropout(config.dropout)
        self.decoder_layernorm = nn.LayerNorm(config.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.decoder_attention_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.dropout,
            activation=config.activation_function,
            attn_dropout=config.attention_dropout,
            act_dropout=config.activation_dropout,
            normalize_before=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)

    def forward(
        self,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        encoder_output: Union[Tuple[Tensor], ModelOutput, None] = None,
        memory_mask: Optional[Tensor] = None,
        cache: Optional[List[Tuple[Cache, StaticCache]]] = None,
        x: Optional[Tensor] = None,
        mix_ratio: Optional[float] = 0,
    ):
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
            x (Tensor, optional):
                The synthetic decoder input embedding of SSTIA strategy.
                Its data type should be `float32` and it has a shape of [batch_size, sequence_length, hidden_size].
                Defaults to `None`, which means don't use SSTIA strategy.
            mix_ratio (float, optional):
                The mixing ratio of synthetic decoder embedding and general deocder input embedding.
                If SSTIA strategy is used, this arg should be set in (0,1).
                Defaults to `0`, which means don't use synthetic decoder embedding.


        Returns:
            Tensor: Returns tensor `decoder_output`, which is the output at the last layer of the model.
            Its data type should be float32 and has a shape of [batch_size, sequence_length, hidden_size].

        """
        if decoder_attention_mask is None:
            decoder_length = paddle.shape(decoder_input_ids)[-1]
            decoder_attention_mask = paddle.tensor.triu(
                (paddle.full((decoder_length, decoder_length), -np.inf, dtype=paddle.get_default_dtype())), 1
            )

        if x is None:
            decoder_inputs_embeds = self.embed_tokens(decoder_input_ids) * self.embed_scale
        else:
            decoder_inputs_embeds = self.embed_tokens(
                decoder_input_ids
            ) * self.embed_scale * mix_ratio + self.embed_scale * x * (1 - mix_ratio)

        past_key_values_length = paddle.shape(cache[0][0].k)[2] if cache is not None else 0
        decoder_inputs_embed_pos = self.decoder_embed_positions(
            paddle.shape(decoder_input_ids), past_key_values_length
        )
        hidden_states = decoder_inputs_embeds + decoder_inputs_embed_pos
        decoder_input = self.decoder_dropout(hidden_states)

        decoder_output = self.decoder(
            tgt=decoder_input,
            memory=encoder_output,
            tgt_mask=decoder_attention_mask,
            memory_mask=memory_mask,
            cache=cache,
        )
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
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`PegasusConfig`):
            An instance of PegasusConfig used to construct PegasusModel.
    """

    def __init__(self, config: PegasusConfig):
        super().__init__(config)
        self.init_std = config.init_std
        self.pad_token_id = config.pad_token_id
        self.decoder_start_token_id = config.decoder_start_token_id
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = PegasusEncoder(config, self.shared)
        self.decoder = PegasusDecoder(config, self.shared)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        encoder_output: Union[Tuple[Tensor], ModelOutput, None] = None,
        use_cache: Optional[bool] = None,
        cache: Optional[List[Tuple[Cache, StaticCache]]] = None,
    ):
        r"""
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

                tokenizer = PegasusTokenizer.from_pretrained(pegasus_path)
                model = PegasusModel.from_pretrained(pegasus_path)

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        """
        if input_ids is None and encoder_output is None:
            raise ValueError("You have to specify either input_ids or encoder_output")
        if decoder_input_ids is None:
            assert input_ids is not None, "input_ids should be " "specified when generating decoder_input_ids"
            decoder_input_ids = shift_tokens_right(input_ids, self.pad_token_id, self.decoder_start_token_id)
        if attention_mask is None:
            assert input_ids is not None, "input_ids should be " "specified when generating attention_mask"
            attention_mask = (
                paddle.cast(input_ids == self.pad_token_id, dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e4
            )
        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4
            attention_mask.stop_gradient = True
        if encoder_output is None:
            encoder_output = self.encoder(input_ids, attention_mask)
        if decoder_attention_mask is not None and decoder_attention_mask.ndim == 2:
            decoder_attention_mask = paddle.unsqueeze(decoder_attention_mask, axis=[1, 2]).astype(
                paddle.get_default_dtype()
            )
            decoder_attention_mask = (1.0 - decoder_attention_mask) * -1e4
            decoder_attention_mask.stop_gradient = True

        if use_cache:
            if cache is None:
                cache = self.decoder.decoder.gen_cache(encoder_output)
        else:
            cache = None

        memory_mask = attention_mask
        if attention_mask is not None:
            if attention_mask.ndim == 4:
                memory_mask = attention_mask[:, :, -1:, :]
            elif attention_mask.ndim == 3:
                memory_mask = attention_mask[:, -1:, :].unsqueeze([1])
            elif attention_mask.ndim == 2:
                memory_mask = attention_mask.unsqueeze([1, 2])
            else:
                raise ValueError("Invalid attention mask shape. ")

        decoder_output, new_cache = self.decoder(
            decoder_input_ids, decoder_attention_mask, encoder_output, memory_mask, cache
        )
        return decoder_output, new_cache, encoder_output, attention_mask


class PegasusForConditionalGeneration(PegasusPretrainedModel):
    r"""
    Pegasus Model with a `language modeling` head on top.

    Args:
        config (:class:`PegasusConfig`):
            An instance of PegasusConfig used to construct PegasusForConditionalGeneration.
    """

    def __init__(self, config: PegasusConfig):
        super().__init__(config)
        self.pegasus = PegasusModel(config)
        self.lm_head_weight = self.create_parameter(
            shape=[config.vocab_size, config.d_model],
            dtype=self.pegasus.shared.weight.dtype,
            is_bias=False,
        )
        if hasattr(self, "final_logits_bias") and "final_logits_bias" not in self._buffers:
            self.final_logits_bias = paddle.zeros((1, config.vocab_size))
        else:
            self.register_buffer("final_logits_bias", paddle.zeros((1, config.vocab_size)))
        self.use_SSTIA = False
        self.mix_ratio = 0

    def get_encoder(self):
        return self.pegasus.get_encoder()

    def get_decoder(self):
        return self.pegasus.get_decoder()

    def prepare_fast_entry(self, kwargs):
        from paddlenlp.ops import FasterPegasus

        decode_strategy = kwargs.get("decode_strategy")
        use_fp16_decoding = kwargs.get("use_fp16_decoding", False)
        decoding_lib = kwargs.get("decoding_lib", None)
        enable_fast_encoder = kwargs.get("enable_fast_encoder", True)
        if decode_strategy == "sampling" and kwargs.get("top_k") != 0 and kwargs.get("top_p") != 1:
            raise AttributeError(
                "Only topk sampling or topp sampling are supported. "
                "Topk sampling and topp sampling cannot be both applied in the fast version."
            )
        if kwargs["repetition_penalty"] != 1.0:
            # not support for repetition_penalty yet in the fast version
            raise AttributeError("'repetition_penalty != 1' is not supported yet in the fast version")
        self._fast_entry = FasterPegasus(
            self,
            use_fp16_decoding=use_fp16_decoding,
            decoding_lib=decoding_lib,
            enable_fast_encoder=enable_fast_encoder,
        ).forward
        return self._fast_entry

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        encoder_output: Union[Tuple[Tensor], ModelOutput, None] = None,
        use_cache: Optional[bool] = None,
        cache: Optional[List[Tuple[Cache, StaticCache]]] = None,
        labels: Optional[Tensor] = None,
    ):
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

                tokenizer = PegasusTokenizer.from_pretrained(pegasus_path)
                model = PegasusForConditionalGeneration.from_pretrained(pegasus_path)

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

        """
        output, new_cache, encoder_output, attention_mask = self.pegasus(
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, encoder_output, use_cache, cache
        )
        lm_logits = paddle.tensor.matmul(output, self.lm_head_weight, transpose_y=True) + self.final_logits_bias

        if self.use_SSTIA:
            assert 0 < self.mix_ratio < 1
            x = lm_logits.clone()
            length = len(x[0])
            for idx in range(length - 1, -1, -1):
                x[:, idx] = x[:, idx - 1]
            x[:, 0, 0] = 2 * paddle.max(x[:, 0])
            x = paddle.nn.functional.softmax(x, axis=2)

            with paddle.no_grad():
                embed_matrix = self.pegasus.decoder.embed_tokens.weight.clone()
                decoder_in = paddle.einsum("blv,ve->ble", x, embed_matrix)

            output_new, _ = self.pegasus.decoder(
                decoder_input_ids,
                decoder_attention_mask,
                encoder_output,
                attention_mask,
                cache,
                x=decoder_in,
                mix_ratio=self.mix_ratio,
            )
            lm_logits_new = (
                paddle.tensor.matmul(output_new, self.lm_head_weight, transpose_y=True) + self.final_logits_bias
            )

            masked_lm_loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                masked_lm_loss = (1 - self.mix_ratio) * loss_fct(
                    lm_logits.reshape((-1, self.pegasus.config["vocab_size"])), labels.reshape((-1,))
                )
                masked_lm_loss += self.mix_ratio * loss_fct(
                    lm_logits_new.reshape((-1, self.pegasus.config["vocab_size"])), labels.reshape((-1,))
                )
                p = paddle.nn.functional.log_softmax(lm_logits_new, axis=2)
                q = paddle.nn.functional.softmax(lm_logits, axis=2)
                loss_kl = paddle.nn.functional.kl_div(p, q, reduction="mean")
                masked_lm_loss += loss_kl
            return lm_logits, new_cache, masked_lm_loss

        else:
            masked_lm_loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                masked_lm_loss = loss_fct(
                    lm_logits.reshape((-1, self.pegasus.config["vocab_size"])), labels.reshape((-1,))
                )

            return lm_logits, new_cache, masked_lm_loss

    def prepare_decoder_input_ids_from_labels(self, labels):
        return shift_tokens_right(labels, self.pegasus.pad_token_id, self.pegasus.config["decoder_start_token_id"])

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
                decoder_attention_mask = decoder_attention_mask[:, :, -1, :].unsqueeze(2)

        return {
            "input_ids": None,
            "decoder_input_ids": decoder_input_ids,
            "encoder_output": encoder_output,
            "decoder_attention_mask": decoder_attention_mask,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "cache": cache,
        }

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(getattr(self, self.base_model_prefix), name)
