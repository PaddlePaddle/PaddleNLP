# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The Fairseq Authors, Microsoft Research, and the HuggingFace Inc. team. All rights reserved.
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

""" PyTorch SpeechT5 model."""

import math
import random
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.utils import recompute
from paddle.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss

from ...utils.initializer import (
    constant_,
    kaiming_normal_,
    normal_,
    ones_,
    uniform_,
    zeros_,
)
from ...utils.log import logger
from ..activations import ACT2FN
from ..model_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqSpectrogramOutput,
)
from ..model_utils import PretrainedModel

# from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration import SpeechT5Config, SpeechT5HifiGanConfig

_HIDDEN_STATES_START_POSITION = 1

# General docstring
_CONFIG_FOR_DOC = "SpeechT5Config"


SPEECHT5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/speecht5_asr",
    "microsoft/speecht5_tts",
    "microsoft/speecht5_vc",
    # See all SpeechT5 models at https://huggingface.co/models?filter=speecht5
]


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


def finfo(dtype: paddle.dtype = None):
    if dtype is None:
        dtype = paddle.get_default_dtype()

    if dtype == paddle.bfloat16:
        # Numpy do not support `np.finfo(np.uint16)`, so try to construct a finfo object to fetch min value
        class BFloatFInfo:
            min = -3.3895313892515355e38

        return BFloatFInfo
    if dtype == paddle.float32:
        return np.finfo(np.float32)
    if dtype == paddle.float16:
        return np.finfo(np.float16)
    if dtype == paddle.float64:
        return np.finfo(np.float64)


def Parameter(tensor):
    return paddle.create_parameter(tensor.shape, dtype=tensor.dtype, default_initializer=nn.initializer.Assign(tensor))


# Copied from paddlenlp.transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: paddle.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = paddle.zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    masked_fill(shifted_input_ids, shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def shift_spectrograms_right(input_values: paddle.Tensor, reduction_factor: int = 1):
    """
    Shift input spectrograms one timestep to the right. Also applies the reduction factor to the sequence length.
    """
    # thin out frames for reduction factor
    if reduction_factor > 1:
        input_values = input_values[:, reduction_factor - 1 :: reduction_factor]

    shifted_input_values = paddle.zeros(input_values.shape)
    shifted_input_values[:, 1:] = input_values[:, :-1].clone()

    # replace possible -100 values in labels by zeros
    masked_fill(shifted_input_values, shifted_input_values == -100.0, 0.0)

    return shifted_input_values


# Copied from paddlenlp.transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: paddle.shape, dtype: paddle.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = paddle.full((tgt_len, tgt_len), float(finfo(dtype).min))
    mask_cond = paddle.arange(mask.shape[-1])
    masked_fill(mask, mask_cond < (mask_cond + 1).reshape([mask.shape[-1], 1]), 0)
    mask = mask.cast(dtype)

    if past_key_values_length > 0:
        mask = paddle.concat([paddle.zeros([tgt_len, past_key_values_length], dtype=dtype), mask], axis=-1)
    return mask[None, None, :, :].expand([bsz, 1, tgt_len, tgt_len + past_key_values_length])


# Copied from paddlenlp.transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: paddle.Tensor, dtype: paddle.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand([bsz, 1, tgt_len, src_len]).cast(dtype)

    inverted_mask = 1.0 - expanded_mask

    return masked_fill(inverted_mask, inverted_mask.cast("bool"), finfo(dtype).min)


# Copied from paddlenlp.transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[paddle.Tensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape([batch_size, max_num_masked_span * mask_length])

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        [batch_size, max_num_masked_span * mask_length]
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask


# Copied from paddlenlp.transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2NoLayerNormConvLayer with Wav2Vec2->SpeechT5
class SpeechT5NoLayerNormConvLayer(nn.Layer):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1D(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias_attr=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from paddlenlp.transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2LayerNormConvLayer with Wav2Vec2->SpeechT5
class SpeechT5LayerNormConvLayer(nn.Layer):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1D(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias_attr=config.conv_bias,
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)

        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from paddlenlp.transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GroupNormConvLayer with Wav2Vec2->SpeechT5
class SpeechT5GroupNormConvLayer(nn.Layer):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1D(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias_attr=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]

        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from paddlenlp.transformers.models.speech_to_text.modeling_speech_to_text.Speech2TextSinusoidalPositionalEmbedding with Speech2Text->SpeechT5
class SpeechT5SinusoidalPositionalEmbedding(nn.Layer):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # in forward put the weights on the correct dtype and device of the param
            emb_weights = emb_weights.cast(dtype=self.weights.dtype)

        self.weights = Parameter(emb_weights)
        self.weights.stop_gradient = True
        self.weights.detach()

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = paddle.exp(paddle.arange(half_dim, dtype="float32") * -emb)
        emb = paddle.arange(num_embeddings, dtype="float32").unsqueeze(1) * emb.unsqueeze(0)
        emb = paddle.concat([paddle.sin(emb), paddle.cos(emb)], axis=1).reshape([num_embeddings, -1])
        if embedding_dim % 2 == 1:
            # zero pad
            emb = paddle.concat([emb, paddle.zeros([num_embeddings, 1])], axis=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb.cast(paddle.get_default_dtype())

    @paddle.no_grad()
    def forward(self, input_ids: paddle.Tensor, past_key_values_length: int = 0):
        bsz, seq_len = input_ids.shape
        # Create the position ids from the input token ids. Any padded tokens remain padded.
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)

        # expand embeddings if needed
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.shape[0]:
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        return self.weights.index_select(axis=0, index=position_ids.reshape([-1])).reshape([bsz, seq_len, -1]).detach()

    def create_position_ids_from_input_ids(
        self, input_ids: paddle.Tensor, padding_idx: int, past_key_values_length: Optional[int] = 0
    ):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: paddle.Tensor x:
        Returns: paddle.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        # mask = input_ids.ne(padding_idx).cast('int32')
        mask = input_ids.cast("int64").not_equal(paddle.to_tensor([padding_idx], dtype="int64")).cast("int32")
        incremental_indices = (paddle.cumsum(mask, axis=1).cast(mask.dtype) + past_key_values_length) * mask
        return incremental_indices.cast("int64") + padding_idx


# Copied from paddlenlp.transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding with Wav2Vec2->SpeechT5
class SpeechT5PositionalConvEmbedding(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1D(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )
        # self.conv = nn.utils.weight_norm(self.conv, name="weight")
        self.padding = SpeechT5SamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):

        hidden_states = hidden_states.transpose([0, 2, 1])

        hidden_states = self.conv(hidden_states)
        print(self.conv.weight)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose([0, 2, 1])
        return hidden_states


class SpeechT5ScaledPositionalEncoding(nn.Layer):
    """
    Scaled positional encoding, see §3.2 in https://arxiv.org/abs/1809.08895
    """

    def __init__(self, dropout, dim, max_len=5000):
        pe = paddle.zeros([max_len, dim])
        position = paddle.arange(0, max_len).unsqueeze(1)
        div_term = paddle.exp((paddle.arange(0, dim, 2, dtype="float32") * -(math.log(10000.0) / dim)))
        pe[:, 0::2] = paddle.sin(position.cast("float32") * div_term)
        pe[:, 1::2] = paddle.cos(position.cast("float32") * div_term)
        pe = pe.unsqueeze(0)
        super().__init__()
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim
        self.alpha = Parameter(paddle.to_tensor([1.0]))

    def forward(self, emb):
        emb = emb + self.alpha * self.pe[:, : emb.shape[1]]
        emb = self.dropout(emb)
        return emb


class SpeechT5RelativePositionalEncoding(nn.Layer):
    def __init__(self, dim, max_length=1000):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        self.pe_k = paddle.nn.Embedding(2 * max_length, dim)

    def forward(self, hidden_states):
        seq_len = hidden_states.shape[1]
        pos_seq = paddle.arange(0, seq_len).cast("int64")
        pos_seq = pos_seq[:, None] - pos_seq[None, :]

        pos_seq[pos_seq < -self.max_length] = -self.max_length
        pos_seq[pos_seq >= self.max_length] = self.max_length - 1
        pos_seq = pos_seq + self.max_length

        return self.pe_k(pos_seq)


# Copied from paddlenlp.transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer with Wav2Vec2->SpeechT5
class SpeechT5SamePadLayer(nn.Layer):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


# Copied from paddlenlp.transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder with Wav2Vec2->SpeechT5
class SpeechT5FeatureEncoder(nn.Layer):
    """Construct the features from raw audio waveform"""

    def __init__(self, config):
        super().__init__()

        if config.feat_extract_norm == "group":
            conv_layers = [SpeechT5GroupNormConvLayer(config, layer_id=0)] + [
                SpeechT5NoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                SpeechT5LayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = nn.LayerList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        for param in self.parameters():
            param.stop_gradient = True
        self._requires_grad = False

    def forward(self, input_values):
        hidden_states = input_values[:, None]

        # make sure hidden_states require grad for gradient_checkpointing
        if self._requires_grad and self.training:
            hidden_states.stop_gradiet = False

        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = recompute(
                    create_custom_forward(conv_layer),
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states.cast("float32"))

        return hidden_states


# Copied from paddlenlp.transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureProjection with Wav2Vec2->SpeechT5
class SpeechT5FeatureProjection(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], epsilon=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


class SpeechT5SpeechEncoderPrenet(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_encoder = SpeechT5FeatureEncoder(config)
        self.feature_projection = SpeechT5FeatureProjection(config)

        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = Parameter(uniform_(paddle.to_tensor([config.hidden_size], dtype="float32"), 0, 1))

        self.pos_conv_embed = SpeechT5PositionalConvEmbedding(config)
        self.pos_sinusoidal_embed = SpeechT5SinusoidalPositionalEmbedding(
            config.max_speech_positions + config.pad_token_id + 1,
            config.hidden_size,
            config.pad_token_id,
        )

    def freeze_feature_encoder(self):
        self.feature_encoder._freeze_parameters()

    def forward(
        self,
        input_values: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        mask_time_indices: Optional[paddle.Tensor] = None,
    ):
        extract_features = self.feature_encoder(input_values)
        extract_features = extract_features.transpose([0, 2, 1])
        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1],
                attention_mask,
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )
        positional_conv_embedding = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + positional_conv_embedding

        if attention_mask is not None:
            padding_mask = attention_mask.not_equal(paddle.to_tensor([1], dtype="bool")).cast("int64")
        else:
            padding_mask = paddle.zeros(hidden_states.shape[:2], dtype="int64")

        positional_sinusoidal_embeddings = self.pos_sinusoidal_embed(padding_mask)
        hidden_states = hidden_states + positional_sinusoidal_embeddings
        return hidden_states, attention_mask

    # Copied from paddlenlp.transformers.models.unispeech.modeling_unispeech.UniSpeechPretrainedModel._get_feature_vector_attention_mask
    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: paddle.Tensor):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(axis=-1)[:, -1]
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths).cast("int64")
        batch_size = attention_mask.shape[0]

        attention_mask = paddle.zeros((batch_size, feature_vector_length), dtype=attention_mask.dtype)
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(paddle.arange(attention_mask.shape[0]), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).cast("bool")
        return attention_mask

    # Copied from paddlenlp.transformers.models.unispeech.modeling_unispeech.UniSpeechPretrainedModel._get_feat_extract_output_lengths
    def _get_feat_extract_output_lengths(self, input_lengths: Union[paddle.Tensor, int]):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/paddle.nn.Conv1D.html
            # return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            if isinstance(input_lengths, paddle.Tensor):
                input_lengths = input_lengths.cast("int64")
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    # Copied from paddlenlp.transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model._mask_hidden_states
    def _mask_hidden_states(
        self,
        hidden_states: paddle.Tensor,
        mask_time_indices: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.shape

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.cast(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = paddle.to_tensor(mask_time_indices, dtype="bool")
            hidden_states[mask_time_indices] = self.masked_spec_embed.cast(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = paddle.to_tensor(mask_feature_indices, dtype="bool")
            mask_feature_indices = mask_feature_indices[:, None].expand([-1, sequence_length, -1])
            hidden_states[mask_feature_indices] = 0

        return hidden_states


class SpeechT5SpeechDecoderPrenet(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layers = nn.LayerList(
            [
                nn.Linear(
                    config.num_mel_bins if i == 0 else config.speech_decoder_prenet_units,
                    config.speech_decoder_prenet_units,
                )
                for i in range(config.speech_decoder_prenet_layers)
            ]
        )

        self.final_layer = nn.Linear(config.speech_decoder_prenet_units, config.hidden_size)

        self.encode_positions = SpeechT5ScaledPositionalEncoding(
            config.positional_dropout,
            config.hidden_size,
            config.max_speech_positions,
        )

        self.speaker_embeds_layer = nn.Linear(config.speaker_embedding_dim + config.hidden_size, config.hidden_size)

    def forward(
        self,
        input_values: paddle.Tensor,
        speaker_embeddings: Optional[paddle.Tensor] = None,
    ):
        # Dropout is always applied, even when evaluating. See §2.2 in https://arxiv.org/abs/1712.05884.

        inputs_embeds = input_values
        for layer in self.layers:
            inputs_embeds = nn.functional.relu(layer(inputs_embeds))
            inputs_embeds = nn.functional.dropout(
                inputs_embeds, self.config.speech_decoder_prenet_dropout, training=True
            )

        inputs_embeds = self.final_layer(inputs_embeds)
        inputs_embeds = self.encode_positions(inputs_embeds)

        if speaker_embeddings is not None:
            speaker_embeddings = nn.functional.normalize(speaker_embeddings)
            speaker_embeddings = speaker_embeddings.unsqueeze(1)
            speaker_embeddings = speaker_embeddings.expand([-1, inputs_embeds.shape[1], -1])
            inputs_embeds = paddle.concat([inputs_embeds, speaker_embeddings], axis=-1)
            inputs_embeds = nn.functional.relu(self.speaker_embeds_layer(inputs_embeds))

        return inputs_embeds


class SpeechT5BatchNormConvLayer(nn.Layer):
    def __init__(self, config, layer_id=0):
        super().__init__()

        if layer_id == 0:
            in_conv_dim = config.num_mel_bins
        else:
            in_conv_dim = config.speech_decoder_postnet_units

        if layer_id == config.speech_decoder_postnet_layers - 1:
            out_conv_dim = config.num_mel_bins
        else:
            out_conv_dim = config.speech_decoder_postnet_units

        self.conv = nn.Conv1D(
            in_conv_dim,
            out_conv_dim,
            kernel_size=config.speech_decoder_postnet_kernel,
            stride=1,
            padding=(config.speech_decoder_postnet_kernel - 1) // 2,
            bias_attr=False,
        )
        self.batch_norm = nn.BatchNorm1D(out_conv_dim)

        if layer_id < config.speech_decoder_postnet_layers - 1:
            self.activation = nn.Tanh()
        else:
            self.activation = None

        self.dropout = nn.Dropout(config.speech_decoder_postnet_dropout)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        if self.activation is not None:
            hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SpeechT5SpeechDecoderPostnet(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.feat_out = nn.Linear(config.hidden_size, config.num_mel_bins * config.reduction_factor)
        self.prob_out = nn.Linear(config.hidden_size, config.reduction_factor)

        self.layers = nn.LayerList(
            [SpeechT5BatchNormConvLayer(config, i) for i in range(config.speech_decoder_postnet_layers)]
        )

    def forward(self, hidden_states: paddle.Tensor):
        outputs_before_postnet = self.feat_out(hidden_states).reshape(
            [hidden_states.shape[0], -1, self.config.num_mel_bins]
        )
        outputs_after_postnet = self.postnet(outputs_before_postnet)
        logits = self.prob_out(hidden_states).reshape([hidden_states.shape[0], -1])
        return outputs_before_postnet, outputs_after_postnet, logits

    def postnet(self, hidden_states: paddle.Tensor):
        layer_output = hidden_states.transpose([0, 2, 1])
        for layer in self.layers:
            layer_output = layer(layer_output)
        return hidden_states + layer_output.transpose([0, 2, 1])


class SpeechT5TextEncoderPrenet(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encode_positions = SpeechT5ScaledPositionalEncoding(
            config.positional_dropout,
            config.hidden_size,
            config.max_text_positions,
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids: paddle.Tensor):
        inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = self.encode_positions(inputs_embeds)
        return inputs_embeds


class SpeechT5TextDecoderPrenet(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.positional_dropout)
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        self.embed_positions = SpeechT5SinusoidalPositionalEmbedding(
            config.max_text_positions + config.pad_token_id + 1,
            config.hidden_size,
            config.pad_token_id,
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
    ):
        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.reshape([-1, input_shape[-1]])
        else:
            raise ValueError("You have to specify `decoder_input_ids`")

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        positions = self.embed_positions(input_ids, past_key_values_length)

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        inputs_embeds += positions
        inputs_embeds = self.dropout(inputs_embeds)

        return inputs_embeds, attention_mask


class SpeechT5TextDecoderPostnet(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)

    def forward(self, hidden_states: paddle.Tensor):
        return self.lm_head(hidden_states)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings


class SpeechT5Attention(nn.Layer):
    """
    Multi-headed attention from 'Attention Is All You Need' paper with relative position bias (see
    https://aclanthology.org/N18-2074.pdf)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)

    def _shape(self, tensor: paddle.Tensor, seq_len: int, bsz: int):
        return tensor.reshape([bsz, seq_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])

    def forward(
        self,
        hidden_states: paddle.Tensor,
        key_value_states: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        layer_head_mask: Optional[paddle.Tensor] = None,
        position_bias: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = paddle.concat([past_key_value[0], key_states], axis=2)
            value_states = paddle.concat([past_key_value[1], value_states], axis=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(paddle.Tensor, paddle.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(paddle.Tensor, paddle.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).reshape(proj_shape)
        key_states = key_states.reshape(proj_shape)
        value_states = value_states.reshape(proj_shape)

        src_len = key_states.shape[1]
        attn_weights = paddle.bmm(query_states, key_states.transpose([0, 2, 1]))

        if attn_weights.shape != [bsz * self.num_heads, tgt_len, src_len]:
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape}"
            )

        # relative attention bias
        if position_bias is not None:
            reshape_q = query_states.reshape([bsz * self.num_heads, -1, self.head_dim]).transpose([1, 0, 2])
            rel_pos_bias = paddle.matmul(reshape_q, position_bias.transpose([0, 2, 1]))
            rel_pos_bias = rel_pos_bias.transpose([1, 0, 2]).reshape(
                [bsz * self.num_heads, position_bias.shape[0], position_bias.shape[1]]
            )
            attn_weights += rel_pos_bias

        if attention_mask is not None:
            if attention_mask.shape != [bsz, 1, tgt_len, src_len]:
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights.reshape([bsz, self.num_heads, tgt_len, src_len]) + attention_mask
            attn_weights = attn_weights.reshape([bsz * self.num_heads, tgt_len, src_len])

        attn_weights = nn.functional.softmax(attn_weights, axis=-1)

        if layer_head_mask is not None:
            if layer_head_mask.shape != [
                self.num_heads,
            ]:
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.shape}"
                )
            attn_weights = layer_head_mask.reshape([1, -1, 1, 1]) * attn_weights.reshape(
                [bsz, self.num_heads, tgt_len, src_len]
            )
            attn_weights = attn_weights.reshape([bsz * self.num_heads, tgt_len, src_len])

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.reshape([bsz, self.num_heads, tgt_len, src_len])
            attn_weights = attn_weights_reshaped.reshape([bsz * self.num_heads, tgt_len, src_len])
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = paddle.bmm(attn_probs, value_states)
        if attn_output.shape != [bsz * self.num_heads, tgt_len, self.head_dim]:
            raise ValueError(
                f"`attn_output` should be of size {[bsz, self.num_heads, tgt_len, self.head_dim]}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.reshape([bsz, self.num_heads, tgt_len, self.head_dim])
        attn_output = attn_output.transpose([0, 2, 1, 3])

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape([bsz, tgt_len, self.embed_dim])

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class SpeechT5FeedForward(nn.Layer):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(config.hidden_size, intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.output_dense = nn.Linear(intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class SpeechT5EncoderLayer(nn.Layer):
    def __init__(self, config: SpeechT5Config):
        super().__init__()
        self.attention = SpeechT5Attention(
            embed_dim=config.hidden_size,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.feed_forward = SpeechT5FeedForward(config, config.encoder_ffn_dim)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        layer_head_mask: Optional[paddle.Tensor] = None,
        position_bias: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`paddle.Tensor`):
                input to the layer of shape `(batch, seq_len, hidden_size)`
            attention_mask (`paddle.Tensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
                large negative values.
            layer_head_mask (`paddle.Tensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            position_bias (`paddle.Tensor`):
                relative position embeddings of size `(seq_len, seq_len, hidden_size // encoder_attention_heads)`
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )

        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SpeechT5DecoderLayer(nn.Layer):
    def __init__(self, config: SpeechT5Config):
        super().__init__()
        self.self_attn = SpeechT5Attention(
            embed_dim=config.hidden_size,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

        self.encoder_attn = SpeechT5Attention(
            config.hidden_size,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

        self.feed_forward = SpeechT5FeedForward(config, config.decoder_ffn_dim)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        encoder_attention_mask: Optional[paddle.Tensor] = None,
        layer_head_mask: Optional[paddle.Tensor] = None,
        cross_attn_layer_head_mask: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, hidden_size)`
            attention_mask (`paddle.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`paddle.Tensor`):
                cross attention input to the layer of shape `(batch, seq_len, hidden_size)`
            encoder_attention_mask (`paddle.Tensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`paddle.Tensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`paddle.Tensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(paddle.Tensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = self.dropout(hidden_states)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class SpeechT5PretrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SpeechT5Config
    base_model_prefix = "speecht5"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, SpeechT5PositionalConvEmbedding):

            normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv._kernel_size[0] * module.conv._in_channels)),
            )
            constant_(module.conv.bias, 0)
        elif isinstance(module, SpeechT5FeatureProjection):
            # module.projection.weight.shape[0] == module.projection.in_features
            k = math.sqrt(1 / module.projection.weight.shape[0])
            uniform_(module.projection.weight, a=-k, b=k)
            uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            zeros_(module.bias)
            ones_(module.weight)
        elif isinstance(module, nn.Conv1D):
            kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module._groups / (module._in_channels * module._kernel_size[0]))
                uniform_(module.bias, a=-k, b=k)
        elif isinstance(module, nn.Embedding):
            normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module._padding_idx is not None:
                zeros_(module.weight[module._padding_idx])

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (SpeechT5Encoder, SpeechT5Decoder, SpeechT5FeatureEncoder)):
            module.gradient_checkpointing = value


class SpeechT5Encoder(SpeechT5PretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* layers. Each layer is a [`SpeechT5EncoderLayer`].
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layerdrop = config.encoder_layerdrop

        self.layers = nn.LayerList([SpeechT5EncoderLayer(config) for _ in range(config.encoder_layers)])

        self.embed_positions = SpeechT5RelativePositionalEncoding(
            config.hidden_size // config.encoder_attention_heads, config.encoder_max_relative_position
        )

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Args:
            hidden_states (`paddle.Tensor` of shape `(batch_size, sequence_length, feature_size)`):
                Features extracted from the speech or text input by the encoder prenet.
            attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing convolution and attention on padding token indices. Mask values selected in
                `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            head_mask (`paddle.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        position_bias = self.embed_positions(hidden_states)

        # deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.shape[0] != len(self.layers):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.shape[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)
            # print(dropout_probability)

            skip_the_layer = self.training and (dropout_probability < self.layerdrop)
            if not skip_the_layer:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = recompute(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        position_bias,
                    )
                else:

                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_bias=position_bias,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class SpeechT5EncoderWithSpeechPrenet(SpeechT5PretrainedModel):
    """
    Wrapper around SpeechT5Encoder that applies SpeechT5SpeechEncoderPrenet to convert the audio waveform data to
    hidden features.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        self.prenet = SpeechT5SpeechEncoderPrenet(config)
        self.wrapped_encoder = SpeechT5Encoder(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
        self,
        input_values: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        hidden_states, attention_mask = self.prenet(input_values, attention_mask)

        outputs = self.wrapped_encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs


class SpeechT5EncoderWithTextPrenet(SpeechT5PretrainedModel):
    """
    Wrapper around SpeechT5Encoder that applies SpeechT5TextEncoderPrenet to convert the input_ids to hidden features.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        self.prenet = SpeechT5TextEncoderPrenet(config)
        self.wrapped_encoder = SpeechT5Encoder(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.init_weights()

    def get_input_embeddings(self):
        return self.prenet.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.prenet.set_input_embeddings(value)

    def forward(
        self,
        input_values: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        hidden_states = self.prenet(input_values)

        outputs = self.wrapped_encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs


class SpeechT5EncoderWithoutPrenet(SpeechT5PretrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when used in combination with
    [`SpeechT5Model`].
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        self.wrapped_encoder = SpeechT5Encoder(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
        self,
        input_values: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        return self.wrapped_encoder(
            hidden_states=input_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class SpeechT5Decoder(SpeechT5PretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`SpeechT5DecoderLayer`]
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        self.layerdrop = config.decoder_layerdrop

        self.layers = nn.LayerList([SpeechT5DecoderLayer(config) for _ in range(config.decoder_layers)])

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.init_weights()

    # Copied from paddlenlp.transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        encoder_attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        cross_attn_head_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            hidden_states (`paddle.Tensor` of shape `(batch_size, sequence_length, feature_size)`):
                Features extracted from the speech or text input by the decoder prenet.
            attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`paddle.Tensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`paddle.Tensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`paddle.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`paddle.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(paddle.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(paddle.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`paddle.Tensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_shape = hidden_states.shape[:-1]

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, hidden_states, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, hidden_states.dtype, tgt_len=input_shape[-1])

        # deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.shape[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.shape[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)

            skip_the_layer = self.training and (dropout_probability < self.layerdrop)
            if skip_the_layer:
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = recompute(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class SpeechT5DecoderWithSpeechPrenet(SpeechT5PretrainedModel):
    """
    Wrapper around SpeechT5Decoder that applies SpeechT5SpeechDecoderPrenet to convert log-mel filterbanks to hidden
    features.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        self.prenet = SpeechT5SpeechDecoderPrenet(config)
        self.wrapped_decoder = SpeechT5Decoder(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
        self,
        input_values: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        encoder_attention_mask: Optional[paddle.Tensor] = None,
        speaker_embeddings: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        cross_attn_head_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        decoder_hidden_states = self.prenet(input_values, speaker_embeddings)

        outputs = self.wrapped_decoder(
            hidden_states=decoder_hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs


class SpeechT5DecoderWithTextPrenet(SpeechT5PretrainedModel):
    """
    Wrapper around SpeechT5Decoder that applies SpeechT5TextDecoderPrenet to convert input tokens to hidden features.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        self.prenet = SpeechT5TextDecoderPrenet(config)
        self.wrapped_decoder = SpeechT5Decoder(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.init_weights()

    def get_input_embeddings(self):
        return self.prenet.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.prenet.set_input_embeddings(value)

    def forward(
        self,
        input_values: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        encoder_attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        cross_attn_head_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        decoder_hidden_states, attention_mask = self.prenet(input_values, attention_mask, past_key_values)

        outputs = self.wrapped_decoder(
            hidden_states=decoder_hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs


class SpeechT5DecoderWithoutPrenet(SpeechT5PretrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when used in combination with
    [`SpeechT5Model`].
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        self.wrapped_decoder = SpeechT5Decoder(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
        self,
        input_values: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        encoder_attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        cross_attn_head_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        outputs = self.wrapped_decoder(
            hidden_states=input_values,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs


class SpeechT5GuidedMultiheadAttentionLoss(nn.Layer):
    """
    Guided attention loss from the paper [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional
    Networks with Guided Attention](https://arxiv.org/abs/1710.08969), adapted for multi-head attention.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__()
        self.sigma = config.guided_attention_loss_sigma
        self.scale = config.guided_attention_loss_scale

    def forward(
        self, attentions: paddle.Tensor, input_masks: paddle.Tensor, output_masks: paddle.Tensor
    ) -> paddle.Tensor:
        """
        Compute the attention loss.

        Args:
            attentions (`paddle.Tensor` of shape `(batch_size, layers * heads, output_sequence_length, input_sequence_length)`):
                Batch of multi-head attention weights
            input_masks (`paddle.Tensor` of shape `(batch_size, input_sequence_length)`):
                Input attention mask as booleans.
            output_masks (`paddle.Tensor` of shape `(batch_size, output_sequence_length)`):
                Target attention mask as booleans.

        Returns:
            `paddle.Tensor` with the loss value
        """
        guided_attn_masks = self._make_guided_attention_masks(input_masks, output_masks)
        masks = output_masks.unsqueeze(-1) & input_masks.unsqueeze(-2)
        masks = masks.unsqueeze(1)

        losses = guided_attn_masks * attentions
        loss = paddle.mean(losses.masked_select(masks))
        return self.scale * loss

    def _make_guided_attention_masks(self, input_masks, output_masks):
        input_lengths = input_masks.sum(-1)
        output_lengths = output_masks.sum(-1)

        guided_attn_masks = paddle.zeros((len(input_masks), output_masks.shape[1], input_masks.shape[1]))

        for idx, (ilen, olen) in enumerate(zip(input_lengths, output_lengths)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(ilen, olen, self.sigma)

        return guided_attn_masks.unsqueeze(1)

    @staticmethod
    def _make_guided_attention_mask(input_length, output_length, sigma):
        grid_y, grid_x = paddle.meshgrid(
            paddle.arange(input_length),
            paddle.arange(output_length),
            indexing="xy",
        )
        grid_x = grid_x.cast("float32") / output_length
        grid_y = grid_y.cast("float32") / input_length
        return 1.0 - paddle.exp(-((grid_y - grid_x) ** 2) / (2 * (sigma**2)))


class SpeechT5SpectrogramLoss(nn.Layer):
    """
    Loss computation used by SpeechT5ForTextToSpeech.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__()
        self.use_guided_attention_loss = config.use_guided_attention_loss
        self.guided_attention_loss_num_heads = config.guided_attention_loss_num_heads
        self.reduction_factor = config.reduction_factor

        self.l1_criterion = L1Loss()
        self.bce_criterion = BCEWithLogitsLoss(pos_weight=paddle.to_tensor([5.0]))

        if self.use_guided_attention_loss:
            self.attn_criterion = SpeechT5GuidedMultiheadAttentionLoss(config)

    def forward(
        self,
        attention_mask: paddle.Tensor,
        outputs_before_postnet: paddle.Tensor,
        outputs_after_postnet: paddle.Tensor,
        logits: paddle.Tensor,
        labels: paddle.Tensor,
        cross_attentions: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        padding_mask = labels != -100.0

        # mask out the padded portions
        labels = labels.masked_select(padding_mask)
        outputs_before_postnet = outputs_before_postnet.masked_select(padding_mask)
        outputs_after_postnet = outputs_after_postnet.masked_select(padding_mask)

        # spectrogram loss
        l1_loss = self.l1_criterion(outputs_after_postnet, labels) + self.l1_criterion(outputs_before_postnet, labels)

        # construct stop labels from the padding mask
        masks = padding_mask[:, :, 0]
        stop_labels = paddle.concat([~masks * 1.0, paddle.ones(masks.shape[0], 1)], axis=1)
        stop_labels = stop_labels[:, 1:].masked_select(masks)
        logits = logits.masked_select(masks)

        # stop token loss
        bce_loss = self.bce_criterion(logits, stop_labels)

        # combined loss
        loss = l1_loss + bce_loss

        # guided attention loss
        if self.use_guided_attention_loss:
            attn = paddle.concat([x[:, : self.guided_attention_loss_num_heads] for x in cross_attentions], axis=1)
            input_masks = attention_mask == 1
            output_masks = padding_mask[:, :, 0]
            if self.reduction_factor > 1:
                output_masks = output_masks[:, self.reduction_factor - 1 :: self.reduction_factor]
            attn_loss = self.attn_criterion(attn, input_masks, output_masks)
            loss += attn_loss

        return loss


class SpeechT5Model(SpeechT5PretrainedModel):
    def __init__(
        self,
        config: SpeechT5Config,
        encoder: Optional[nn.Layer] = None,
        decoder: Optional[nn.Layer] = None,
    ):
        super().__init__(config)
        self.config = config
        self.encoder = SpeechT5EncoderWithoutPrenet(config) if encoder is None else encoder
        self.decoder = SpeechT5DecoderWithoutPrenet(config) if decoder is None else decoder

        # Initialize weights and apply final processing
        self.init_weights()

    def get_input_embeddings(self):
        if isinstance(self.encoder, SpeechT5EncoderWithTextPrenet):
            return self.encoder.get_input_embeddings()
        if isinstance(self.decoder, SpeechT5DecoderWithTextPrenet):
            return self.decoder.get_input_embeddings()
        return None

    def set_input_embeddings(self, value):
        if isinstance(self.encoder, SpeechT5EncoderWithTextPrenet):
            self.encoder.set_input_embeddings(value)
        if isinstance(self.decoder, SpeechT5DecoderWithTextPrenet):
            self.decoder.set_input_embeddings(value)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        if isinstance(self.encoder, SpeechT5EncoderWithSpeechPrenet):
            self.encoder.prenet.freeze_feature_encoder()

    def forward(
        self,
        input_values: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        decoder_input_values: Optional[paddle.Tensor] = None,
        decoder_attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        decoder_head_mask: Optional[paddle.Tensor] = None,
        cross_attn_head_mask: Optional[paddle.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        speaker_embeddings: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[paddle.Tensor], Seq2SeqModelOutput]:
        r"""
        input_values (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
            Depending on which encoder is being used, the `input_values` are either: float values of the input raw
            speech waveform, or indices of input sequence tokens in the vocabulary, or hidden states.

        decoder_input_values (`paddle.Tensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Depending on which decoder is being used, the `decoder_input_values` are either: float values of log-mel
            filterbank features extracted from the raw speech waveform, or indices of decoder input sequence tokens in
            the vocabulary, or hidden states.

        speaker_embeddings (`paddle.Tensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*):
            Tensor containing the speaker embeddings.

        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_values=input_values,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # downsample encoder attention mask (only for encoders with speech input)
        if attention_mask is not None and isinstance(self.encoder, SpeechT5EncoderWithSpeechPrenet):
            encoder_attention_mask = self.encoder.prenet._get_feature_vector_attention_mask(
                encoder_outputs[0].shape[1], attention_mask
            )
        else:
            encoder_attention_mask = attention_mask

        if isinstance(self.decoder, SpeechT5DecoderWithSpeechPrenet):
            decoder_args = {"speaker_embeddings": speaker_embeddings}
        else:
            decoder_args = {}

        decoder_outputs = self.decoder(
            input_values=decoder_input_values,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **decoder_args,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class SpeechT5ForSpeechToText(SpeechT5PretrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"speecht5.encoder.prenet.pos_sinusoidal_embed.weights",
        r"text_decoder_postnet.lm_head.weight",
    ]
    _keys_to_ignore_on_save = [
        r"speecht5.encoder.prenet.pos_sinusoidal_embed.weights",
    ]

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that does not define the"
                " vocabulary size of the language model head. Please instantiate the model as follows:"
                " `SpeechT5ForSpeechToText.from_pretrained(..., vocab_size=vocab_size)`. or define `vocab_size` of"
                " your model's configuration."
            )

        speech_encoder = SpeechT5EncoderWithSpeechPrenet(config)
        text_decoder = SpeechT5DecoderWithTextPrenet(config)
        self.speecht5 = SpeechT5Model(config, speech_encoder, text_decoder)

        self.text_decoder_postnet = SpeechT5TextDecoderPostnet(config)

        # Initialize weights and apply final processing
        self.init_weights()

    def get_encoder(self):
        return self.speecht5.get_encoder()

    def get_decoder(self):
        return self.speecht5.get_decoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.get_encoder().prenet.freeze_feature_encoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        return new_embeddings

    def get_output_embeddings(self):
        return self.text_decoder_postnet.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.text_decoder_postnet.set_output_embeddings(new_embeddings)

    def forward(
        self,
        input_values: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        decoder_input_ids: Optional[paddle.Tensor] = None,
        decoder_attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        decoder_head_mask: Optional[paddle.Tensor] = None,
        cross_attn_head_mask: Optional[paddle.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        input_values (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
            into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (*pip install
            soundfile*). To prepare the array into `input_values`, the [`SpeechT5Processor`] should be used for padding
            and conversion into a tensor of type `paddle.Tensor`. See [`SpeechT5Processor.__call__`] for details.

        decoder_input_ids (`paddle.Tensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`SpeechT5Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            SpeechT5 uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If
            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

        labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            Label indices can be obtained using [`SpeechT5Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

        Returns:

        Example:

        ```python
        >>> from paddlenlp.transformers import SpeechT5Processor, SpeechT5ForSpeechToText
        >>> from datasets import load_dataset

        >>> dataset = load_dataset(
        ...     "hf-internal-testing/librispeech_asr_demo", "clean", split="validation"
        ... )  # doctest: +IGNORE_RESULT
        >>> dataset = dataset.sort("id")
        >>> sampling_rate = dataset.features["audio"].sampling_rate

        >>> processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")
        >>> model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr")

        >>> # audio file is decoded on the fly
        >>> inputs = processor(audio=dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pd")
        >>> predicted_ids = model.generate(**inputs, max_length=100)

        >>> # transcribe speech
        >>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        >>> transcription[0]
        'mister quilter is the apostle of the middle classes and we are glad to welcome his gospel'
        ```

        ```python
        >>> inputs["labels"] = processor(text_target=dataset[0]["text"], return_tensors="pd").input_ids

        >>> # compute loss
        >>> loss = model(**inputs).loss
        >>> round(loss.item(), 2)
        19.68
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        outputs = self.speecht5(
            input_values=input_values,
            attention_mask=attention_mask,
            decoder_input_values=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = self.text_decoder_postnet(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape([-1, self.config.vocab_size]), labels.reshape([-1]))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


def _generate_speech(
    model: SpeechT5PretrainedModel,
    input_values: paddle.Tensor,
    speaker_embeddings: Optional[paddle.Tensor] = None,
    threshold: float = 0.5,
    minlenratio: float = 0.0,
    maxlenratio: float = 20.0,
    vocoder: Optional[nn.Layer] = None,
    output_cross_attentions: bool = False,
) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
    encoder_attention_mask = paddle.ones_like(input_values)

    encoder_out = model.speecht5.encoder(
        input_values=input_values,
        attention_mask=encoder_attention_mask,
        return_dict=True,
    )

    encoder_last_hidden_state = encoder_out.last_hidden_state

    # downsample encoder attention mask
    if isinstance(model.speecht5.encoder, SpeechT5EncoderWithSpeechPrenet):
        encoder_attention_mask = model.speecht5.encoder.prenet._get_feature_vector_attention_mask(
            encoder_out[0].shape[1], encoder_attention_mask
        )

    maxlen = int(encoder_last_hidden_state.shape[1] * maxlenratio / model.config.reduction_factor)
    minlen = int(encoder_last_hidden_state.shape[1] * minlenratio / model.config.reduction_factor)

    # Start the output sequence with a mel spectrum that is all zeros.
    output_sequence = paddle.zeros([1, 1, model.config.num_mel_bins], dtype=encoder_last_hidden_state.dtype)

    spectrogram = []
    cross_attentions = []
    past_key_values = None
    idx = 0

    while True:
        idx += 1

        # Run the decoder prenet on the entire output sequence.
        decoder_hidden_states = model.speecht5.decoder.prenet(output_sequence, speaker_embeddings)

        # Run the decoder layers on the last element of the prenet output.
        decoder_out = model.speecht5.decoder.wrapped_decoder(
            hidden_states=decoder_hidden_states[:, -1:],
            attention_mask=None,
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=output_cross_attentions,
            return_dict=True,
        )

        if output_cross_attentions:
            cross_attentions.append(paddle.concat(decoder_out.cross_attentions, axis=0))

        last_decoder_output = decoder_out.last_hidden_state[0, -1]
        past_key_values = decoder_out.past_key_values

        # Predict the new mel spectrum for this step in the sequence.
        spectrum = model.speech_decoder_postnet.feat_out(last_decoder_output)
        spectrum = spectrum.reshape([model.config.reduction_factor, model.config.num_mel_bins])
        spectrogram.append(spectrum)

        # Extend the output sequence with the new mel spectrum.
        output_sequence = paddle.concat(
            (output_sequence, spectrum[-1].reshape([1, 1, model.config.num_mel_bins])), axis=1
        )

        # Predict the probability that this is the stop token.
        prob = F.sigmoid(model.speech_decoder_postnet.prob_out(last_decoder_output))

        # Finished when stop token or maximum length is reached.
        if idx >= minlen and (int(sum(prob.numpy() >= threshold)) > 0 or idx >= maxlen):
            spectrogram = paddle.concat(spectrogram, axis=0).unsqueeze(0)
            spectrogram = model.speech_decoder_postnet.postnet(spectrogram)
            spectrogram = spectrogram.squeeze(0)
            break

    if vocoder is not None:
        outputs = vocoder(spectrogram)
    else:
        outputs = spectrogram

    if output_cross_attentions:
        cross_attentions = paddle.concat(cross_attentions, axis=2)
        outputs = (outputs, cross_attentions)

    return outputs


class SpeechT5ForTextToSpeech(SpeechT5PretrainedModel):
    _keys_to_ignore_on_load_missing = []
    _keys_to_ignore_on_save = []

    main_input_name = "input_ids"

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that does not define the"
                " vocabulary size of the language model head. Please instantiate the model as follows:"
                " `SpeechT5ForTextToSpeech.from_pretrained(..., vocab_size=vocab_size)`. or define `vocab_size` of"
                " your model's configuration."
            )

        text_encoder = SpeechT5EncoderWithTextPrenet(config)
        speech_decoder = SpeechT5DecoderWithSpeechPrenet(config)
        self.speecht5 = SpeechT5Model(config, text_encoder, speech_decoder)

        self.speech_decoder_postnet = SpeechT5SpeechDecoderPostnet(config)

        # Initialize weights and apply final processing
        self.init_weights()

    def get_encoder(self):
        return self.speecht5.get_encoder()

    def get_decoder(self):
        return self.speecht5.get_decoder()

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        decoder_input_values: Optional[paddle.Tensor] = None,
        decoder_attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        decoder_head_mask: Optional[paddle.Tensor] = None,
        cross_attn_head_mask: Optional[paddle.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        speaker_embeddings: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        stop_labels: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, Seq2SeqSpectrogramOutput]:
        r"""
        input_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. The `batch_size` should be 1 currently.

            Indices can be obtained using [`SpeechT5Tokenizer`]. See [`~PreTrainedTokenizer.encode`] and
            [`~PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        decoder_input_values (`paddle.Tensor` of shape `(batch_size, sequence_length, config.num_mel_bins)`):
            Float values of input mel spectrogram.

            SpeechT5 uses an all-zero spectrum as the starting token for `decoder_input_values` generation. If
            `past_key_values` is used, optionally only the last `decoder_input_values` have to be input (see
            `past_key_values`).
        speaker_embeddings (`paddle.Tensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*):
            Tensor containing the speaker embeddings.
        labels (`paddle.Tensor` of shape `(batch_size, sequence_length, config.num_mel_bins)`, *optional*):
            Float values of target mel spectrogram. Timesteps set to `-100.0` are ignored (masked) for the loss
            computation. Spectrograms can be obtained using [`SpeechT5Processor`]. See [`SpeechT5Processor.__call__`]
            for details.

        Returns:

        Example:

        ```python
        >>> from paddlenlp.transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, set_seed
        >>> import paddle

        >>> processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        >>> model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        >>> vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        >>> inputs = processor(text="Hello, my dog is cute", return_tensors="pd")
        >>> speaker_embeddings = paddle.zeros((1, 512))  # or load xvectors from a file

        >>> set_seed(555)  # make deterministic

        >>> # generate speech
        >>> speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        >>> speech.shape
            [15872]
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if stop_labels is not None:
            warnings.warn(
                "The argument `stop_labels` is deprecated and will be removed in version 4.30.0 of Transformers",
                FutureWarning,
            )

        if labels is not None:
            if decoder_input_values is None:
                decoder_input_values = shift_spectrograms_right(labels, self.config.reduction_factor)
            if self.config.use_guided_attention_loss:
                output_attentions = True

        outputs = self.speecht5(
            input_values=input_ids,
            attention_mask=attention_mask,
            decoder_input_values=decoder_input_values,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            speaker_embeddings=speaker_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        outputs_before_postnet, outputs_after_postnet, logits = self.speech_decoder_postnet(outputs[0])

        loss = None
        if labels is not None:
            criterion = SpeechT5SpectrogramLoss(self.config)
            loss = criterion(
                attention_mask,
                outputs_before_postnet,
                outputs_after_postnet,
                logits,
                labels,
                outputs.cross_attentions,
            )

        if not return_dict:
            output = (outputs_after_postnet,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSpectrogramOutput(
            loss=loss,
            spectrogram=outputs_after_postnet,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    @paddle.no_grad()
    def generate_speech(
        self,
        input_ids: paddle.Tensor,
        speaker_embeddings: Optional[paddle.Tensor] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 20.0,
        vocoder: Optional[nn.Layer] = None,
        output_cross_attentions: bool = False,
    ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
        r"""
        Converts a sequence of input tokens into a sequence of mel spectrograms, which are subsequently turned into a
        speech waveform using a vocoder.

        Args:
            input_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. The `batch_size` should be 1 currently.

                Indices can be obtained using [`SpeechT5Tokenizer`]. See [`~PreTrainedTokenizer.encode`] and
                [`~PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            speaker_embeddings (`paddle.Tensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*):
                Tensor containing the speaker embeddings.
            threshold (`float`, *optional*, defaults to 0.5):
                The generated sequence ends when the predicted stop token probability exceeds this value.
            minlenratio (`float`, *optional*, defaults to 0.0):
                Used to calculate the minimum required length for the output sequence.
            maxlenratio (`float`, *optional*, defaults to 20.0):
                Used to calculate the maximum allowed length for the output sequence.
            vocoder (`nn.Layer`, *optional*, defaults to `None`):
                The vocoder that converts the mel spectrogram into a speech waveform. If `None`, the output is the mel
                spectrogram.
            output_cross_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of the decoder's cross-attention layers.

        Returns:
            `tuple(paddle.Tensor)` comprising various elements depending on the inputs:
            - **spectrogram** (*optional*, returned when no `vocoder` is provided) `paddle.Tensor` of shape
              `(output_sequence_length, config.num_mel_bins)` -- The predicted log-mel spectrogram.
            - **waveform** (*optional*, returned when a `vocoder` is provided) `paddle.Tensor` of shape
              `(num_frames,)` -- The predicted speech waveform.
            - **cross_attentions** (*optional*, returned when `output_cross_attentions` is `True`) `paddle.Tensor`
              of shape `(config.decoder_layers, config.decoder_attention_heads, output_sequence_length,
              input_sequence_length)` -- The outputs of the decoder's cross-attention layers.
        """
        return _generate_speech(
            self,
            input_ids,
            speaker_embeddings,
            threshold,
            minlenratio,
            maxlenratio,
            vocoder,
            output_cross_attentions,
        )


class SpeechT5ForSpeechToSpeech(SpeechT5PretrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"speecht5.encoder.prenet.pos_sinusoidal_embed.weights",
    ]
    _keys_to_ignore_on_save = [
        r"speecht5.encoder.prenet.pos_sinusoidal_embed.weights",
    ]

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)

        speech_encoder = SpeechT5EncoderWithSpeechPrenet(config)
        speech_decoder = SpeechT5DecoderWithSpeechPrenet(config)
        self.speecht5 = SpeechT5Model(config, speech_encoder, speech_decoder)

        self.speech_decoder_postnet = SpeechT5SpeechDecoderPostnet(config)

        # Initialize weights and apply final processing
        self.init_weights()

    def get_encoder(self):
        return self.speecht5.get_encoder()

    def get_decoder(self):
        return self.speecht5.get_decoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.get_encoder().prenet.freeze_feature_encoder()

    def forward(
        self,
        input_values: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        decoder_input_values: Optional[paddle.Tensor] = None,
        decoder_attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        decoder_head_mask: Optional[paddle.Tensor] = None,
        cross_attn_head_mask: Optional[paddle.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        speaker_embeddings: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        stop_labels: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, Seq2SeqSpectrogramOutput]:
        r"""
        input_values (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
            into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (*pip install
            soundfile*). To prepare the array into `input_values`, the [`SpeechT5Processor`] should be used for padding
            and conversion into a tensor of type `paddle.Tensor`. See [`SpeechT5Processor.__call__`] for details.
        decoder_input_values (`paddle.Tensor` of shape `(batch_size, sequence_length, config.num_mel_bins)`):
            Float values of input mel spectrogram.

            SpeechT5 uses an all-zero spectrum as the starting token for `decoder_input_values` generation. If
            `past_key_values` is used, optionally only the last `decoder_input_values` have to be input (see
            `past_key_values`).
        speaker_embeddings (`paddle.Tensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*):
            Tensor containing the speaker embeddings.
        labels (`paddle.Tensor` of shape `(batch_size, sequence_length, config.num_mel_bins)`, *optional*):
            Float values of target mel spectrogram. Spectrograms can be obtained using [`SpeechT5Processor`]. See
            [`SpeechT5Processor.__call__`] for details.

        Returns:

        Example:

        ```python
        >>> from paddlenlp.transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan, set_seed
        >>> from datasets import load_dataset
        >>> import paddle

        >>> dataset = load_dataset(
        ...     "hf-internal-testing/librispeech_asr_demo", "clean", split="validation"
        ... )  # doctest: +IGNORE_RESULT
        >>> dataset = dataset.sort("id")
        >>> sampling_rate = dataset.features["audio"].sampling_rate

        >>> processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
        >>> model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
        >>> vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        >>> # audio file is decoded on the fly
        >>> inputs = processor(audio=dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pd")

        >>> speaker_embeddings = paddle.zeros((1, 512))  # or load xvectors from a file

        >>> set_seed(555)  # make deterministic

        >>> # generate speech
        >>> speech = model.generate_speech(inputs["input_values"], speaker_embeddings, vocoder=vocoder)
        >>> speech.shape
        [77824]
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if stop_labels is not None:
            warnings.warn(
                "The argument `stop_labels` is deprecated and will be removed in version 4.30.0 of Transformers",
                FutureWarning,
            )

        if labels is not None:
            if decoder_input_values is None:
                decoder_input_values = shift_spectrograms_right(labels, self.config.reduction_factor)

        outputs = self.speecht5(
            input_values=input_values,
            attention_mask=attention_mask,
            decoder_input_values=decoder_input_values,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            speaker_embeddings=speaker_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        _, spectrogram, logits = self.speech_decoder_postnet(outputs[0])

        loss = None

        if not return_dict:
            output = (spectrogram,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSpectrogramOutput(
            loss=loss,
            spectrogram=spectrogram,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    @paddle.no_grad()
    def generate_speech(
        self,
        input_values: paddle.Tensor,
        speaker_embeddings: Optional[paddle.Tensor] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 20.0,
        vocoder: Optional[nn.Layer] = None,
        output_cross_attentions: bool = False,
    ) -> paddle.Tensor:
        r"""
        Converts a raw speech waveform into a sequence of mel spectrograms, which are subsequently turned back into a
        speech waveform using a vocoder.

        Args:
            input_values (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
                Float values of input raw speech waveform. The `batch_size` should be 1 currently.

                Values can be obtained by loading a *.flac* or *.wav* audio file into an array of type `List[float]` or
                a `numpy.ndarray`, *e.g.* via the soundfile library (*pip install soundfile*). To prepare the array
                into `input_values`, the [`SpeechT5Processor`] should be used for padding and conversion into a tensor
                of type `paddle.Tensor`. See [`SpeechT5Processor.__call__`] for details.
            speaker_embeddings (`paddle.Tensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*):
                Tensor containing the speaker embeddings.
            threshold (`float`, *optional*, defaults to 0.5):
                The generated sequence ends when the predicted stop token probability exceeds this value.
            minlenratio (`float`, *optional*, defaults to 0.0):
                Used to calculate the minimum required length for the output sequence.
            maxlenratio (`float`, *optional*, defaults to 20.0):
                Used to calculate the maximum allowed length for the output sequence.
            vocoder (`nn.Layer`, *optional*, defaults to `None`):
                The vocoder that converts the mel spectrogram into a speech waveform. If `None`, the output is the mel
                spectrogram.
            output_cross_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of the decoder's cross-attention layers.

        Returns:
            `tuple(paddle.Tensor)` comprising various elements depending on the inputs:
            - **spectrogram** (*optional*, returned when no `vocoder` is provided) `paddle.Tensor` of shape
              `(output_sequence_length, config.num_mel_bins)` -- The predicted log-mel spectrogram.
            - **waveform** (*optional*, returned when a `vocoder` is provided) `paddle.Tensor` of shape
              `(num_frames,)` -- The predicted speech waveform.
            - **cross_attentions** (*optional*, returned when `output_cross_attentions` is `True`) `paddle.Tensor`
              of shape `(config.decoder_layers, config.decoder_attention_heads, output_sequence_length,
              input_sequence_length)` -- The outputs of the decoder's cross-attention layers.
        """
        if speaker_embeddings is None:
            speaker_embeddings = paddle.zeros((1, 512))

        return _generate_speech(
            self,
            input_values,
            speaker_embeddings,
            threshold,
            minlenratio,
            maxlenratio,
            vocoder,
            output_cross_attentions,
        )


class HifiGanResidualBlock(nn.Layer):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        self.convs1 = nn.LayerList(
            [
                nn.Conv1D(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation[i],
                    padding=self.get_padding(kernel_size, dilation[i]),
                )
                for i in range(len(dilation))
            ]
        )
        self.convs2 = nn.LayerList(
            [
                nn.Conv1D(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=self.get_padding(kernel_size, 1),
                )
                for _ in range(len(dilation))
            ]
        )

    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

    def apply_weight_norm(self):
        for layer in self.convs1:
            nn.utils.weight_norm(layer)
        for layer in self.convs2:
            nn.utils.weight_norm(layer)

    def remove_weight_norm(self):
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)

    def forward(self, hidden_states):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv1(hidden_states)
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv2(hidden_states)
            hidden_states = hidden_states + residual
        return hidden_states


class SpeechT5HifiGan(PretrainedModel):
    config_class = SpeechT5HifiGanConfig
    main_input_name = "spectrogram"

    def __init__(self, config: SpeechT5HifiGanConfig):
        super().__init__(config)
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = nn.Conv1D(
            config.model_in_dim,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.upsampler = nn.LayerList()
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.upsampler.append(
                nn.Conv1DTranspose(
                    config.upsample_initial_channel // (2**i),
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel_size,
                    stride=upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2,
                )
            )

        self.resblocks = nn.LayerList()
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))

        self.conv_post = nn.Conv1D(channels, 1, kernel_size=7, stride=1, padding=3)

        self.register_buffer("mean", paddle.zeros([config.model_in_dim]))
        self.register_buffer("scale", paddle.ones([config.model_in_dim]))

        # Initialize weights and apply final processing
        self.init_weights()

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1D)):
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # module.bias.data.zero_()
                zeros_(module.bias)

    def apply_weight_norm(self):
        nn.utils.weight_norm(self.conv_pre)
        for layer in self.upsampler:
            nn.utils.weight_norm(layer)
        for layer in self.resblocks:
            layer.apply_weight_norm()
        nn.utils.weight_norm(self.conv_post)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv_pre)
        for layer in self.upsampler:
            nn.utils.remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        nn.utils.remove_weight_norm(self.conv_post)

    def forward(self, spectrogram: paddle.Tensor) -> paddle.Tensor:
        r"""
        Converts a log-mel spectrogram into a speech waveform. Passing a batch of log-mel spectrograms returns a batch
        of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a single, un-batched speech
        waveform.

        Args:
            spectrogram (`paddle.Tensor`):
                Tensor containing the log-mel spectrograms. Can be batched and of shape `(batch_size, sequence_length,
                config.model_in_dim)`, or un-batched and of shape `(sequence_length, config.model_in_dim)`.

        Returns:
            `paddle.Tensor`: Tensor containing the speech waveform. If the input spectrogram is batched, will be of
            shape `(batch_size, num_frames,)`. If un-batched, will be of shape `(num_frames,)`.
        """
        if self.config.normalize_before:
            spectrogram = (spectrogram - self.mean) / self.scale

        is_batched = spectrogram.dim() == 3
        if not is_batched:
            spectrogram = spectrogram.unsqueeze(0)
        hidden_states = spectrogram.transpose([0, 2, 1])

        hidden_states = self.conv_pre(hidden_states)
        for i in range(self.num_upsamples):
            hidden_states = nn.functional.leaky_relu(hidden_states, self.config.leaky_relu_slope)
            hidden_states = self.upsampler[i](hidden_states)

            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            hidden_states = res_state / self.num_kernels

        hidden_states = nn.functional.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        hidden_states = paddle.tanh(hidden_states)

        if not is_batched:
            # remove batch dim and collapse tensor to 1-d audio waveform
            waveform = hidden_states.squeeze(0).transpose([1, 0]).reshape([-1])
        else:
            # remove seq-len dim since this collapses to 1
            waveform = hidden_states.squeeze(1)

        return waveform
