# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The LAION-AI Team and The HuggingFace Team. All rights reserved.
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

import collections
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.utils import recompute

from paddlenlp.utils.log import logger

from ...utils.initializer import normal_, ones_, zeros_
from ..activations import ACT2FN
from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
    ModelOutput,
)
from ..model_utils import (
    PretrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from .configuration import ClapAudioConfig, ClapConfig, ClapTextConfig

CLAP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "laion/clap-htsat-fused",
    "laion/clap-htsat-unfused",
    # See all clap models at https://huggingface.co/models?filter=clap
]


def Parameter(tensor):
    return paddle.create_parameter(tensor.shape, dtype=tensor.dtype, default_initializer=nn.initializer.Assign(tensor))


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


# Adapted from: https://github.com/LAION-AI/CLAP/blob/6ad05a971ba0622f6acee8c41993e0d02bbed639/src/open_clip/utils.py#L191
def interpolate(hidden_states, ratio):
    """
    Interpolate data in time domain. This is used to compensate the resolution reduction in downsampling of a CNN.

    Args:
        hidden_states (`paddle.Tensor` of shape (batch_size, time_length, classes_num)):
            Input hidden states
        ratio (`int`):
            The ratio of the length of the output to the length of the input.
    """
    (batch_size, time_length, classes_num) = hidden_states.shape
    upsampled = hidden_states[:, :, None, :].tile([1, 1, ratio, 1])
    upsampled = upsampled.reshape([batch_size, time_length * ratio, classes_num])
    return upsampled


# Adapted from https://github.com/LAION-AI/CLAP/blob/6ad05a971ba0622f6acee8c41993e0d02bbed639/src/open_clip/htsat.py#L249
def window_partition(hidden_states, window_size):
    """
    Returns the resized hidden states. The output shape should be `(batch_size * num_windows, window_size, window_size,
    num_channels)`

    Args:
        hidden_states (`paddle.Tensor` of shape `(batch_size, height, width, num_channels)`):
            Input hidden states
        window_size (`int`):
            Window size
    """
    batch_size, height, width, num_channels = hidden_states.shape

    hidden_states = hidden_states.reshape(
        [batch_size, height // window_size, window_size, width // window_size, window_size, num_channels]
    )
    windows = hidden_states.transpose([0, 1, 3, 2, 4, 5]).reshape([-1, window_size, window_size, num_channels])
    return windows


# Adapted from https://github.com/LAION-AI/CLAP/blob/6ad05a971ba0622f6acee8c41993e0d02bbed639/src/open_clip/htsat.py#L263
def window_reverse(windows, window_size, height, width):
    """
    Args:
        windows (`paddle.Tensor` of shape `(num_windows * batch_size, window_size, window_size, num_channels)`):
            Input windows
        window_size (`int`):
            Window size
        height (`int`):
            Height of the resized audio
        width (`int`):
            Width of the resized audio
    """
    batch_size = int(windows.shape[0] / (height * width / window_size / window_size))

    hidden_states = windows.reshape(
        [batch_size, height // window_size, width // window_size, window_size, window_size, -1]
    )
    hidden_states = hidden_states.transpose([0, 1, 3, 2, 4, 5]).reshape([batch_size, height, width, -1])
    return hidden_states


# Copied from paddlenlp.transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: paddle.Tensor x:

    Returns: paddle.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.not_equal(paddle.to_tensor(padding_idx, dtype="int32")).cast("int32")

    incremental_indices = (paddle.cumsum(mask, axis=1).cast(mask.dtype) + past_key_values_length) * mask
    return incremental_indices.cast("int64") + padding_idx


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html#CLIP-loss-function
def contrastive_loss(logits: paddle.Tensor) -> paddle.Tensor:
    labels = paddle.arange(len(logits))
    return nn.functional.cross_entropy(logits, labels)


@dataclass
# Copied from paddlenlp.transformers.models.clip.modeling_clip.CLIPTextModelOutput with CLIP->Clap
class ClapTextModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds (`paddle.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    text_embeds: Optional[paddle.Tensor] = None
    last_hidden_state: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class ClapAudioModelOutput(ModelOutput):
    """
    ClapAudio model output to mimic the output of the original implementation.

    Args:
        audio_embeds (`paddle.Tensor` of shape `(batch_size, hidden_size)`):
            The Audio embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    audio_embeds: Optional[paddle.Tensor] = None
    last_hidden_state: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
# Copied from paddlenlp.transformers.models.clip.modeling_clip.CLIPOutput with CLIP->Clap, vision->audio, Vision->Audio, image->audio
class ClapOutput(ModelOutput):
    """
    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for audio-text similarity.
        logits_per_audio:(`paddle.Tensor` of shape `(audio_batch_size, text_batch_size)`):
            The scaled dot product scores between `audio_embeds` and `text_embeds`. This represents the audio-text
            similarity scores.
        logits_per_text:(`paddle.Tensor` of shape `(text_batch_size, audio_batch_size)`):
            The scaled dot product scores between `text_embeds` and `audio_embeds`. This represents the text-audio
            similarity scores.
        text_embeds(`paddle.Tensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`ClapTextModel`].
        audio_embeds(`paddle.Tensor` of shape `(batch_size, output_dim`):
            The audio embeddings obtained by applying the projection layer to the pooled output of [`ClapAudioModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`ClapTextModel`].
        audio_model_output(`BaseModelOutputWithPooling`):
            The output of the [`ClapAudioModel`].
    """

    loss: Optional[paddle.Tensor] = None
    logits_per_audio: paddle.Tensor = None
    logits_per_text: paddle.Tensor = None
    text_embeds: paddle.Tensor = None
    audio_embeds: paddle.Tensor = None
    text_model_output: BaseModelOutputWithPooling = None
    audio_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "audio_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# Adapted from paddlenlp.transformers.models.swin.modeling_swin.SwinDropPath
class ClapDropPath(nn.Layer):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks). This is a slightly
    refactored version of the `SwinDropPath` implementation.
    """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states):
        if self.drop_prob == 0.0 or not self.training:
            return hidden_states

        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (hidden_states.shape[0],) + (1,) * (hidden_states.ndim - 1)

        random_tensor = keep_prob + paddle.rand(shape, dtype=hidden_states.dtype)
        random_tensor = paddle.floor(random_tensor)  # binarize
        output = (hidden_states / keep_prob) * random_tensor
        return output


# Adapted from https://github.com/LAION-AI/CLAP/blob/6ad05a971ba0622f6acee8c41993e0d02bbed639/src/open_clip/feature_fusion.py#L133
class ClapAudioAFFBlock(nn.Layer):
    r"""
    ATTENTIONAL FEATURE FUSION Block from CLAP, since in CLAP we are always in 2D mode, it is not needed to implement
    the 1D version.
    """

    def __init__(self, config: ClapAudioConfig):
        super().__init__()
        channels = config.patch_embeds_hidden_size
        downsize_ratio = config.aff_block_r
        inter_channels = int(channels // downsize_ratio)

        self.local_att = nn.Sequential(
            nn.Conv2D(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(inter_channels),
            nn.ReLU(),
            nn.Conv2D(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(inter_channels),
            nn.ReLU(),
            nn.Conv2D(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states, residual):
        attention_input = hidden_states + residual

        fused_layer_output = self.local_att(attention_input) + self.global_att(attention_input)
        fused_layer_output = self.sigmoid(fused_layer_output)

        output = 2 * hidden_states * fused_layer_output + 2 * residual * (1 - fused_layer_output)
        return output


class ClapAudioPatchEmbed(nn.Layer):
    """
    This module converts the hidden states reshaped as an image to patch embeddings ready to be passed to the
    Transformer block.
    """

    def __init__(self, config: ClapAudioConfig):
        super().__init__()
        img_size = (config.spec_size, config.spec_size) if isinstance(config.spec_size, int) else config.spec_size
        patch_size = (
            (config.patch_size, config.patch_size) if isinstance(config.patch_size, int) else config.patch_size
        )
        patch_stride = (
            (config.patch_stride, config.patch_stride) if isinstance(config.patch_stride, int) else config.patch_stride
        )

        self.img_size = img_size
        self.patch_stride = patch_stride

        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.flatten = config.flatten_patch_embeds
        self.enable_fusion = config.enable_fusion

        padding = ((patch_size[0] - patch_stride[0]) // 2, (patch_size[1] - patch_stride[1]) // 2)

        scale_factor = 4 if (self.enable_fusion) and (config.fusion_type == "channel_map") else 1

        self.proj = nn.Conv2D(
            config.patch_embed_input_channels * scale_factor,
            config.patch_embeds_hidden_size,
            kernel_size=patch_size,
            stride=patch_stride,
            padding=padding,
        )

        self.norm = nn.LayerNorm(config.patch_embeds_hidden_size) if config.enable_patch_layer_norm else nn.Identity()
        if self.enable_fusion:
            self.fusion_model = ClapAudioAFFBlock(config)
            self.mel_conv2d = nn.Conv2D(
                config.patch_embed_input_channels,
                config.patch_embeds_hidden_size,
                kernel_size=(patch_size[0], patch_size[1] * 3),
                stride=(patch_stride[0], patch_stride[1] * 3),
                padding=padding,
            )

    def forward(self, hidden_states, is_longer_idx=None):
        if self.enable_fusion:
            # retrieve the last mel as we have transposed the input
            global_hidden_states = hidden_states[:, 0:1, :, :]

            # global processing
            batch_size, num_channels, height, width = global_hidden_states.shape

            if height != self.img_size[0] or width != self.img_size[1]:
                raise ValueError(
                    f"Input audio size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
                )

            global_hidden_states = self.proj(global_hidden_states)
            output_width = global_hidden_states.shape[-1]
            if len(is_longer_idx) > 0:
                # local processing
                local_hidden_states = paddle.gather(hidden_states[:, 1:, :, :], is_longer_idx, axis=0)

                batch_size, num_channels, height, width = local_hidden_states.shape
                local_hidden_states = local_hidden_states.reshape([batch_size * num_channels, 1, height, width])

                local_hidden_states = self.mel_conv2d(local_hidden_states)

                _, features, height, width = local_hidden_states.shape
                local_hidden_states = local_hidden_states.reshape([batch_size, num_channels, features, height, width])
                local_hidden_states = local_hidden_states.transpose((0, 2, 3, 1, 4)).flatten(3)

                local_width = local_hidden_states.shape[-1]

                local_hidden_states = nn.functional.pad(
                    local_hidden_states, (0, output_width - local_width, 0, 0), mode="constant", value=0.0
                )

                global_hidden_states[is_longer_idx] = self.fusion_model(
                    paddle.gather(global_hidden_states, is_longer_idx, axis=0), local_hidden_states
                )
            hidden_states = global_hidden_states
        else:
            _, _, height, width = hidden_states.shape
            if height != self.img_size[0] or width != self.img_size[1]:
                raise ValueError(
                    f"Input audio size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
                )
            hidden_states = self.proj(hidden_states)

        if self.flatten:
            hidden_states = hidden_states.flatten(2).transpose([0, 2, 1])
        hidden_states = self.norm(hidden_states)
        return hidden_states


# Copied from paddlenlp.transformers.models.swin.modeling_swin.SwinSelfAttention with Swin->ClapAudio
class ClapAudioSelfAttention(nn.Layer):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )
        self.relative_position_bias_table = Parameter(
            paddle.zeros([(2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads])
        )

        # get pair-wise relative position index for each token inside the window
        coords_h = paddle.arange(self.window_size[0])
        coords_w = paddle.arange(self.window_size[1])
        coords = paddle.stack(paddle.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = paddle.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias_attr=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias_attr=config.qkv_bias)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias_attr=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [self.num_attention_heads, self.attention_head_size]
        x = x.reshape(new_x_shape)
        return x.transpose([0, 2, 1, 3])

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor]:
        batch_size, dim, num_channels = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = paddle.matmul(query_layer, key_layer.transpose([0, 1, 3, 2]))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape([-1])]
        relative_position_bias = relative_position_bias.reshape(
            [self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1]
        )

        relative_position_bias = relative_position_bias.transpose([2, 0, 1])
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in ClapAudioModel forward() function)
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.reshape(
                [batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim]
            )
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.reshape([-1, self.num_attention_heads, dim, dim])

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose([0, 2, 1, 3])
        new_context_layer_shape = context_layer.shape[:-2] + [
            self.all_head_size,
        ]
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from paddlenlp.transformers.models.swin.modeling_swin.SwinSelfOutput with Swin->ClapAudio
class ClapAudioSelfOutput(nn.Layer):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: paddle.Tensor, input_tensor: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from paddlenlp.transformers.models.swin.modeling_swin.SwinAttention with Swin->ClapAudio
class ClapAudioAttention(nn.Layer):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        self.self = ClapAudioSelfAttention(config, dim, num_heads, window_size)
        self.output = ClapAudioSelfOutput(config, dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, axis=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from paddlenlp.transformers.models.swin.modeling_swin.SwinIntermediate with Swin->ClapAudio
class ClapAudioIntermediate(nn.Layer):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from paddlenlp.transformers.models.swin.modeling_swin.SwinOutput with Swin->ClapAudio
class ClapAudioOutput(nn.Layer):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from paddlenlp.transformers.models.swin.modeling_swin.SwinLayer with SwinDropPath->ClapDropPath, Swin->ClapAudio
class ClapAudioLayer(nn.Layer):
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.shift_size = shift_size
        self.window_size = config.window_size
        self.input_resolution = input_resolution
        self.layernorm_before = nn.LayerNorm(dim, epsilon=config.layer_norm_eps)
        self.attention = ClapAudioAttention(config, dim, num_heads, window_size=self.window_size)
        self.drop_path = ClapDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_after = nn.LayerNorm(dim, epsilon=config.layer_norm_eps)
        self.intermediate = ClapAudioIntermediate(config, dim)
        self.output = ClapAudioOutput(config, dim)

    def set_shift_and_window_size(self, input_resolution):
        if min(input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(input_resolution)

    def get_attn_mask(self, height, width, dtype):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = paddle.zeros((1, height, width, 1), dtype=dtype)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.reshape([-1, self.window_size * self.window_size])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = masked_fill(attn_mask, attn_mask != 0, float(-100.0))
            attn_mask = masked_fill(attn_mask, attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def maybe_pad(self, hidden_states, height, width):
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        # (padding_left,padding_right,padding_top, padding_bottom)
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        # hidden_states = nn.functional.pad(hidden_states, pad_values)
        # TODO(wugaosheng): torch pad is different from paddle pad
        hidden_states = nn.functional.pad(hidden_states, (0, pad_right, 0, pad_bottom), data_format="NHWC")

        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: paddle.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        if not always_partition:
            self.set_shift_and_window_size(input_dimensions)
        else:
            pass
        height, width = input_dimensions
        batch_size, _, channels = hidden_states.shape
        shortcut = hidden_states

        hidden_states = self.layernorm_before(hidden_states)

        hidden_states = hidden_states.reshape([batch_size, height, width, channels])

        # pad hidden_states to multiples of window size
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        _, height_pad, width_pad, _ = hidden_states.shape
        # cyclic shift
        if self.shift_size > 0:
            shifted_hidden_states = paddle.roll(
                hidden_states, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2)
            )
        else:
            shifted_hidden_states = hidden_states

        # partition windows
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.reshape([-1, self.window_size * self.window_size, channels])
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)

        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )

        attention_output = attention_outputs[0]

        attention_windows = attention_output.reshape([-1, self.window_size, self.window_size, channels])
        shifted_windows = window_reverse(attention_windows, self.window_size, height_pad, width_pad)

        # reverse cyclic shift
        if self.shift_size > 0:
            attention_windows = paddle.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :]

        attention_windows = attention_windows.reshape([batch_size, height * width, channels])

        hidden_states = shortcut + self.drop_path(attention_windows)

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = hidden_states + self.output(layer_output)

        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs


# Copied from paddlenlp.transformers.models.swin.modeling_swin.SwinStage with Swin->ClapAudio
class ClapAudioStage(nn.Layer):
    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, downsample):
        super().__init__()
        self.config = config
        self.dim = dim
        self.blocks = nn.LayerList(
            [
                ClapAudioLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        self.pointing = False

    def forward(
        self,
        hidden_states: paddle.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor]:
        height, width = input_dimensions
        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition
            )

            hidden_states = layer_outputs[0]

        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


# Copied from paddlenlp.transformers.models.swin.modeling_swin.SwinPatchMerging with Swin->ClapAudio
class ClapAudioPatchMerging(nn.Layer):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Layer`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(self, input_resolution: Tuple[int], dim: int, norm_layer: nn.Layer = nn.LayerNorm) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias_attr=False)
        self.norm = norm_layer(4 * dim)

    def maybe_pad(self, input_feature, height, width):
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, width % 2, 0, height % 2)
            input_feature = nn.functional.pad(input_feature, pad_values, data_format="NHWC")

        return input_feature

    def forward(self, input_feature: paddle.Tensor, input_dimensions: Tuple[int, int]) -> paddle.Tensor:
        height, width = input_dimensions
        # `dim` is height * width
        batch_size, dim, num_channels = input_feature.shape

        input_feature = input_feature.reshape([batch_size, height, width, num_channels])
        # pad input to be disible by width and height, if needed
        input_feature = self.maybe_pad(input_feature, height, width)
        # [batch_size, height/2, width/2, num_channels]
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        # batch_size height/2 width/2 4*num_channels
        input_feature = paddle.concat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
        input_feature = input_feature.reshape([batch_size, -1, 4 * num_channels])  # batch_size height/2*width/2 4*C

        input_feature = self.norm(input_feature)
        input_feature = self.reduction(input_feature)

        return input_feature


class ClapAudioEncoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.num_layers = len(config.depths)

        self.config = config
        self.patch_embed = ClapAudioPatchEmbed(config)
        self.enable_fusion = config.enable_fusion
        self.patch_stride = self.patch_embed.patch_stride
        self.spec_size = config.spec_size
        self.freq_ratio = config.spec_size // config.num_mel_bins

        self.num_features = int(config.patch_embeds_hidden_size * 2 ** (self.num_layers - 1))

        drop_path_rate = [x.item() for x in paddle.linspace(0, config.drop_path_rate, sum(config.depths))]

        grid_size = self.patch_embed.grid_size
        self.input_resolutions = [(grid_size[0] // (2**i), grid_size[1] // (2**i)) for i in range(self.num_layers)]

        self.layers = nn.LayerList(
            [
                ClapAudioStage(
                    config=config,
                    dim=int(config.patch_embeds_hidden_size * 2**i_layer),
                    input_resolution=self.input_resolutions[i_layer],
                    depth=config.depths[i_layer],
                    num_heads=config.num_attention_heads[i_layer],
                    drop_path=drop_path_rate[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                    downsample=ClapAudioPatchMerging if (i_layer < self.num_layers - 1) else None,
                )
                for i_layer in range(self.num_layers)
            ]
        )

        self.gradient_checkpointing = False

        self.batch_norm = nn.BatchNorm2D(config.num_mel_bins, momentum=0.1)
        self.norm = nn.LayerNorm(self.num_features)
        self.depths = config.depths
        self.avgpool = nn.AdaptiveAvgPool1D(1)

    def reshape_mel2img(self, normalized_input_features):
        """
        The input is 4 normalized log mel spectrograms. It is reshape to the common shape of images. Each channel
        should represent 1 of the 4 crops of the spectrogram. For more details, refer to the [`ClapFeatureExtractor`].
        """
        _, _, time_length, freq_length = normalized_input_features.shape

        spec_width = int(self.spec_size * self.freq_ratio)
        spec_heigth = self.spec_size // self.freq_ratio

        if time_length > spec_width or freq_length > spec_heigth:
            raise ValueError("the wav size should be less than or equal to the swin input size")

        # to avoid bicubic zero error
        if time_length < spec_width:
            normalized_input_features = nn.functional.interpolate(
                normalized_input_features, (spec_width, freq_length), mode="bicubic", align_corners=True
            )
        if freq_length < spec_heigth:
            normalized_input_features = nn.functional.interpolate(
                normalized_input_features, (time_length, spec_heigth), mode="bicubic", align_corners=True
            )

        batch, channels, time, freq = normalized_input_features.shape

        # batch_size, channels, spec_width, spec_heigth --> batch_size, channels, spec_heigth * freq_ratio, spec_width // freq_ratio
        normalized_input_features = normalized_input_features.reshape(
            [batch, channels * self.freq_ratio, time // self.freq_ratio, freq]
        )
        normalized_input_features = normalized_input_features.transpose([0, 1, 3, 2])
        normalized_input_features = normalized_input_features.reshape(
            [batch, channels, freq * self.freq_ratio, time // self.freq_ratio]
        )

        return normalized_input_features

    def forward(
        self,
        input_features,
        is_longer: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, ClapAudioModelOutput]:

        input_features = input_features.transpose([0, 3, 2, 1])
        normalized_input_features = self.batch_norm(input_features)
        normalized_input_features = normalized_input_features.transpose([0, 3, 2, 1])
        is_longer_list_idx = None
        if self.enable_fusion:
            is_longer_list = is_longer
            is_longer_list_idx = paddle.where(is_longer_list == 1)[0]

        hidden_states = self.reshape_mel2img(normalized_input_features)

        frames_num = hidden_states.shape[2]

        hidden_states = self.patch_embed(hidden_states, is_longer_list_idx)

        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        input_dimensions = self.input_resolutions[0]

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            # rearrange batch_size (height width) channels -> batch_size channel height width
            reshaped_hidden_state = hidden_states.reshape([batch_size, *input_dimensions, hidden_size])
            reshaped_hidden_state = reshaped_hidden_state.transpose([0, 3, 1, 2])
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            input_dimensions = self.input_resolutions[i]

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = recompute(
                    create_custom_forward(layer_module), hidden_states, input_dimensions, layer_head_mask
                )

            else:
                layer_outputs = layer_module(
                    hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition
                )

            hidden_states = layer_outputs[0]

            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                # rearrange batch_size (height width) channels -> batch_size channel height width
                # here we use the original (not downsampled) height and width
                reshaped_hidden_state = hidden_states_before_downsampling.reshape(
                    [batch_size, *(output_dimensions[0], output_dimensions[1]), hidden_size]
                )
                reshaped_hidden_state = reshaped_hidden_state.transpose([0, 3, 1, 2])
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange batch_size (height width) channels -> batch_size channel height width
                reshaped_hidden_state = hidden_states.reshape([batch_size, *input_dimensions, hidden_size])
                reshaped_hidden_state = reshaped_hidden_state.transpose([0, 3, 1, 2])
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[3:]

        last_hidden_state = self.norm(hidden_states)

        batch_size, _, n_channels = last_hidden_state.shape

        freq_shape = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[0]
        temporal_shape = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]

        last_hidden_state = last_hidden_state.transpose([0, 2, 1]).reshape(
            [batch_size, n_channels, freq_shape, temporal_shape]
        )

        batch_size, n_channels, n_frequencies, n_temp = last_hidden_state.shape
        # group 2D CNN
        c_freq_bin = n_frequencies // self.freq_ratio
        last_hidden_state = last_hidden_state.reshape(
            [batch_size, n_channels, n_frequencies // c_freq_bin, c_freq_bin, n_temp]
        )
        last_hidden_state = last_hidden_state.transpose([0, 1, 3, 2, 4]).reshape(
            [batch_size, n_channels, c_freq_bin, -1]
        )
        latent_output = self.avgpool(paddle.flatten(last_hidden_state, 2))
        latent_output = paddle.flatten(latent_output, 1)

        if not return_dict:
            return tuple(
                v
                for v in [
                    last_hidden_state,
                    latent_output,
                    all_reshaped_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=latent_output,
            hidden_states=all_reshaped_hidden_states,
            attentions=all_self_attentions,
        )


class ClapProjectionLayer(nn.Layer):
    def __init__(self, config: Union[ClapAudioConfig, ClapTextConfig]):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        projection_dim = config.projection_dim

        self.linear1 = nn.Linear(hidden_size, projection_dim)
        self.activation = ACT2FN[config.projection_hidden_act]
        self.linear2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, hidden_states):
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


# Copied from paddlenlp.transformers.models.roberta.modeling_roberta.RobertaEmbeddings with Roberta->ClapText, persistent=False->persistent=True
class ClapTextEmbeddings(nn.Layer):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from paddlenlp.transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", paddle.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer("token_type_ids", paddle.zeros(self.position_ids.shape, dtype="int64"), persistable=True)

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand([input_shape[0], seq_length])
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = paddle.zeros(input_shape, dtype="int64")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: paddle.Tensor

        Returns: paddle.Tensor
        """
        input_shape = inputs_embeds.shape[:-1]
        sequence_length = input_shape[1]

        position_ids = paddle.arange(self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype="int64")
        return position_ids.unsqueeze(0).expand(input_shape)


# Copied from paddlenlp.transformers.models.bert.modeling_bert.BertSelfAttention with Bert->ClapText
class ClapTextSelfAttention(nn.Layer):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: paddle.Tensor) -> paddle.Tensor:
        new_x_shape = x.shape[:-1] + [self.num_attention_heads, self.attention_head_size]
        x = x.reshape(new_x_shape)
        return x.transpose([0, 2, 1, 3])

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        encoder_attention_mask: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = paddle.concat([past_key_value[0], key_layer], axis=2)
            value_layer = paddle.concat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(paddle.Tensor, paddle.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(paddle.Tensor, paddle.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = paddle.matmul(query_layer, key_layer.transpose([0, 1, 3, 2]))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = paddle.to_tensor([key_length - 1], dtype="int64").reshape([-1, 1])
            else:
                position_ids_l = paddle.arange(query_length, dtype="int64").reshape([-1, 1])
            position_ids_r = paddle.arange(key_length, dtype="int64").reshape([1, -1])
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.cast(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = paddle.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = paddle.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = paddle.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in ClapTextModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = paddle.matmul(attention_probs, value_layer)

        context_layer = context_layer.transpose([0, 2, 1, 3])
        new_context_layer_shape = context_layer.shape[:-2] + [
            self.all_head_size,
        ]
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from paddlenlp.transformers.models.bert.modeling_bert.BertSelfOutput
class ClapTextSelfOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: paddle.Tensor, input_tensor: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from paddlenlp.transformers.models.bert.modeling_bert.BertAttention with Bert->ClapText
class ClapTextAttention(nn.Layer):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = ClapTextSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = ClapTextSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, axis=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        encoder_attention_mask: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from paddlenlp.transformers.models.bert.modeling_bert.BertIntermediate
class ClapTextIntermediate(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from paddlenlp.transformers.models.bert.modeling_bert.BertOutput
class ClapTextOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: paddle.Tensor, input_tensor: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from paddlenlp.transformers.models.bert.modeling_bert.BertLayer with Bert->ClapText
class ClapTextLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ClapTextAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = ClapTextAttention(config, position_embedding_type="absolute")
        self.intermediate = ClapTextIntermediate(config)
        self.output = ClapTextOutput(config)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        encoder_attention_mask: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from paddlenlp.transformers.models.bert.modeling_bert.BertEncoder with Bert->ClapText
class ClapTextEncoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.LayerList([ClapTextLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        encoder_attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[paddle.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = recompute(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from paddlenlp.transformers.models.bert.modeling_bert.BertPooler
class ClapTextPooler(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ClapPreTrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ClapConfig
    base_model_prefix = "clap"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids", r"logit_scale_a", r"logit_scale_t"]

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor

        if isinstance(module, ClapTextEmbeddings):
            normal_(module.position_embeddings.weight, mean=0.0, std=factor * 0.02)
            normal_(module.token_type_embeddings.weight, mean=0.0, std=factor * 0.02)
        elif isinstance(module, ClapModel):
            normal_(module.logit_scale_a, std=factor * 0.02)
            normal_(module.logit_scale_t, std=factor * 0.02)
        elif isinstance(module, nn.Embedding):
            normal_(module.weight, mean=0.0, std=factor * 0.02)
        elif isinstance(module, nn.LayerNorm):
            zeros_(module.bias)
            ones_(module.weight)
        elif isinstance(module, (nn.Conv2D, nn.Linear)):
            in_proj_std = (self.config.hidden_size**-0.5) * ((2 * self.config.num_hidden_layers) ** -0.5) * factor
            normal_(module.weight, std=in_proj_std)
            if module.bias is not None:
                zeros_(module.bias)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ClapTextEncoder):
            module.gradient_checkpointing = value

    def get_extended_attention_mask(
        self,
        attention_mask: paddle.Tensor,
        input_shape: Tuple[int],
        has_query: bool = False,
    ) -> paddle.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        Arguments:
            attention_mask (`paddle.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
        Returns:
            `paddle.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - the model is an encoder, so make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.cast(dtype=self.config.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(
        self, head_mask: Optional[paddle.Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> paddle.Tensor:
        """
        Prepare the head mask if needed.
        Args:
            head_mask (`paddle.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.
        Returns:
            `paddle.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.ndim == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand([num_hidden_layers, -1, -1, -1, -1])
        elif head_mask.ndim == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.ndim == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.cast(dtype=self.config.dtype)  # switch to float if need + fp16 compatibility
        return head_mask


class ClapAudioModel(ClapPreTrainedModel):
    config_class = ClapAudioConfig
    main_input_name = "input_features"

    def __init__(self, config: ClapAudioConfig):
        super().__init__(config)
        self.audio_encoder = ClapAudioEncoder(config)
        # Initialize weights and apply final processing
        self.init_weights()

    def get_input_embeddings(self) -> nn.Layer:
        return self.audio_encoder.patch_embed.proj

    def forward(
        self,
        input_features: Optional[paddle.Tensor] = None,
        is_longer: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from paddlenlp.transformers import AutoProcessor, ClapAudioModel

        >>> dataset = load_dataset("ashraq/esc50")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused")
        >>> processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")

        >>> inputs = processor(audios=audio_sample, return_tensors="pd")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return self.audio_encoder(
            input_features=input_features,
            is_longer=is_longer,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class ClapTextModel(ClapPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    config_class = ClapTextConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from paddlenlp.transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->ClapText
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = ClapTextEmbeddings(config)
        self.encoder = ClapTextEncoder(config)

        self.pooler = ClapTextPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # Copied from paddlenlp.transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        encoder_attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[paddle.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(paddle.Tensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = paddle.ones(((batch_size, seq_length + past_key_values_length)))

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand([batch_size, seq_length])
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = paddle.zeros(input_shape, dtype="int64")

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: paddle.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = paddle.ones(encoder_hidden_shape)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class ClapModel(ClapPreTrainedModel):
    config_class = ClapConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: ClapConfig):
        super().__init__(config)

        if not isinstance(config.text_config, ClapTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type ClapTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.audio_config, ClapAudioConfig):
            raise ValueError(
                "config.audio_config is expected to be of type ClapAudioConfig but is of type"
                f" {type(config.audio_config)}."
            )

        text_config = config.text_config
        audio_config = config.audio_config

        self.logit_scale_a = Parameter(paddle.ones([1]) * np.log(config.logit_scale_init_value))
        self.logit_scale_t = Parameter(paddle.ones([1]) * np.log(config.logit_scale_init_value))
        self.projection_dim = config.projection_dim

        self.text_model = ClapTextModel(text_config)
        self.text_projection = ClapProjectionLayer(text_config)

        self.audio_model = ClapAudioModel(audio_config)
        self.audio_projection = ClapProjectionLayer(audio_config)

        # Initialize weights and apply final processing
        self.init_weights()

    def get_text_features(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> paddle.Tensor:
        r"""
        Returns:
            text_features (`paddle.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`ClapTextModel`].

        Examples:

        ```python
        >>> from paddlenlp.transformers import AutoTokenizer, ClapModel

        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        >>> tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

        >>> inputs = tokenizer(["the sound of a cat", "the sound of a dog"], padding=True, return_tensors="pd")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use CLAP model's config for some fields (if specified) instead of those of audio & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1] if return_dict is not None else text_outputs.pooler_output
        text_features = self.text_projection(pooled_output)
        text_features = F.normalize(text_features, axis=-1)

        return text_features

    def get_audio_features(
        self,
        input_features: Optional[paddle.Tensor] = None,
        is_longer: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> paddle.Tensor:
        r"""
        Returns:
            audio_features (`paddle.Tensor` of shape `(batch_size, output_dim`): The audio embeddings obtained by
            applying the projection layer to the pooled output of [`ClapAudioModel`].

        Examples:

        ```python
        >>> from paddlenlp.transformers import AutoFeatureExtractor, ClapModel
        >>> import paddle

        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
        >>> random_audio = paddle.rand((16_000))
        >>> inputs = feature_extractor(random_audio, return_tensors="pd")
        >>> audio_features = model.get_audio_features(**inputs)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        audio_outputs = self.audio_model(
            input_features=input_features,
            is_longer=is_longer,
            return_dict=return_dict,
        )

        pooled_output = audio_outputs[1] if not return_dict else audio_outputs.pooler_output

        audio_features = self.audio_projection(pooled_output)
        audio_features = F.normalize(audio_features, axis=-1)

        return audio_features

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        input_features: Optional[paddle.Tensor] = None,
        is_longer: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ClapOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from paddlenlp.transformers import AutoProcessor, ClapModel
        >>> import paddle.nn.functional as F

        >>> dataset = load_dataset("ashraq/esc50")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        >>> processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

        >>> input_text = ["Sound of a dog", "Sound of vaccum cleaner"]

        >>> inputs = processor(text=input_text, audios=audio_sample, return_tensors="pd", padding=True)

        >>> outputs = model(**inputs)
        >>> logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
        >>> probs = F.softmax(logits_per_audio, axis=-1) # we can take the softmax to get the label probabilities
        ```"""
        # Use CLAP model's config for some fields (if specified) instead of those of audio & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        audio_outputs = self.audio_model(
            input_features=input_features,
            is_longer=is_longer,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        audio_embeds = audio_outputs[1] if not return_dict else audio_outputs.pooler_output
        audio_embeds = self.audio_projection(audio_embeds)

        text_embeds = text_outputs[1] if not return_dict else text_outputs.pooler_output
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        audio_embeds = audio_embeds / audio_embeds.norm(p=2, axis=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, axis=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale_text = self.logit_scale_t.exp()
        logit_scale_audio = self.logit_scale_a.exp()
        logits_per_text = paddle.matmul(text_embeds, audio_embeds.t()) * logit_scale_text
        logits_per_audio = paddle.matmul(audio_embeds, text_embeds.t()) * logit_scale_audio

        loss = None
        if return_loss:
            caption_loss = contrastive_loss(logits_per_text)
            audio_loss = contrastive_loss(logits_per_audio.t())
            loss = (caption_loss + audio_loss) / 2.0

        if not return_dict:
            output = (logits_per_audio, logits_per_text, text_embeds, audio_embeds, text_outputs, audio_outputs)
            return ((loss,) + output) if loss is not None else output

        return ClapOutput(
            loss=loss,
            logits_per_audio=logits_per_audio,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            audio_embeds=audio_embeds,
            text_model_output=text_outputs,
            audio_model_output=audio_outputs,
        )


class ClapTextModelWithProjection(ClapPreTrainedModel):
    config_class = ClapTextConfig

    def __init__(self, config: ClapTextConfig):
        super().__init__(config)
        self.text_model = ClapTextModel(config)
        self.text_projection = ClapProjectionLayer(config)
        # Initialize weights and apply final processing
        self.init_weights()

    def get_input_embeddings(self) -> nn.Layer:
        return self.text_model.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.text_model.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ClapTextModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from paddlenlp.transformers import AutoTokenizer, ClapTextModelWithProjection

        >>> model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-unfused")
        >>> tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

        >>> inputs = tokenizer(["a sound of a cat", "a sound of a dog"], padding=True, return_tensors="pd")

        >>> outputs = model(**inputs)
        >>> text_embeds = outputs.text_embeds
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1] if not return_dict else text_outputs.pooler_output

        text_embeds = self.text_projection(pooled_output)

        if not return_dict:
            outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return ClapTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=text_outputs.last_hidden_state,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )


class ClapAudioModelWithProjection(ClapPreTrainedModel):
    config_class = ClapAudioConfig
    main_input_name = "input_features"

    def __init__(self, config: ClapAudioConfig):
        super().__init__(config)
        self.audio_model = ClapAudioModel(config)
        self.audio_projection = ClapProjectionLayer(config)
        # Initialize weights and apply final processing
        self.init_weights()

    def get_input_embeddings(self) -> nn.Layer:
        return self.audio_model.audio_encoder.patch_embed.proj

    def forward(
        self,
        input_features: Optional[paddle.Tensor] = None,
        is_longer: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ClapAudioModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from paddlenlp.transformers import ClapAudioModelWithProjection, ClapProcessor

        >>> model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused")
        >>> processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

        >>> dataset = load_dataset("ashraq/esc50")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> inputs = processor(audios=audio_sample, return_tensors="pd")
        >>> outputs = model(**inputs)
        >>> audio_embeds = outputs.audio_embeds
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        audio_outputs = self.audio_model(
            input_features=input_features,
            is_longer=is_longer,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = audio_outputs[1] if not return_dict else audio_outputs.pooler_output

        audio_embeds = self.audio_projection(pooled_output)

        if not return_dict:
            outputs = (audio_embeds, audio_outputs[0]) + audio_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return ClapAudioModelOutput(
            audio_embeds=audio_embeds,
            last_hidden_state=audio_outputs.last_hidden_state,
            attentions=audio_outputs.attentions,
            hidden_states=audio_outputs.hidden_states,
        )
