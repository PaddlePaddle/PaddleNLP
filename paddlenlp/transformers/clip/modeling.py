# coding=utf-8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import nn

from ...utils.converter import StateDictNameMapping
from ...utils.initializer import normal_, ones_, zeros_
from ..model_outputs import BaseModelOutputWithPooling, ModelOutput
from ..model_utils import PretrainedModel
from .configuration import CLIPConfig, CLIPTextConfig, CLIPVisionConfig

CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # vit model
    "openai/clip-vit-base-patch32",  # ViT-B/32
    "openai/clip-vit-base-patch16",  # ViT-B/16
    "openai/clip-vit-large-patch14",  # ViT-L/14
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    # resnet model
    "openai/clip-rn50",  # RN50
    "openai/clip-rn101",  # RN101
    "openai/clip-rn50x4",  # RN50x4
]

__all__ = [
    "ModifiedResNet",
    "CLIPVisionTransformer",
    "CLIPTextTransformer",
    "CLIPTextModel",
    "CLIPVisionModel",
    "CLIPPretrainedModel",
    "CLIPModel",
    "CLIPTextModelWithProjection",
    "CLIPVisionModelWithProjection",
]


def quick_gelu(x):
    return x * F.sigmoid(1.702 * x)


F.quick_gelu = quick_gelu

NEG_INF = -1e4  # float("-inf") -1e4 -1e9

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html


def contrastive_loss(logits: paddle.Tensor) -> paddle.Tensor:
    return F.cross_entropy(logits, paddle.arange(len(logits)))


def clip_loss(similarity: paddle.Tensor) -> paddle.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@dataclass
class CLIPVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`paddle.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
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

    image_embeds: Optional[paddle.Tensor] = None
    last_hidden_state: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class CLIPTextModelOutput(ModelOutput):
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
class CLIPOutput(ModelOutput):
    """
    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`paddle.Tensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`paddle.Tensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`paddle.Tensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
        image_embeds(`paddle.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPVisionModel`].
    """

    loss: Optional[paddle.Tensor] = None
    logits_per_image: paddle.Tensor = None
    logits_per_text: paddle.Tensor = None
    text_embeds: paddle.Tensor = None
    image_embeds: paddle.Tensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class ModifiedResNet(nn.Layer):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2D(3, width // 2, kernel_size=3, stride=2, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(width // 2)
        self.conv2 = nn.Conv2D(width // 2, width // 2, kernel_size=3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(width // 2)
        self.conv3 = nn.Conv2D(width // 2, width, kernel_size=3, padding=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(width)
        self.avgpool = nn.AvgPool2D(2)
        self.relu = nn.ReLU()

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


def multi_head_attention_forward(
    x: paddle.Tensor,
    num_heads: int,
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    v_proj: nn.Linear,
    c_proj: nn.Linear,
    attn_mask: Optional[paddle.Tensor] = None,
):
    max_len, batch_size, emb_dim = x.shape
    head_dim = emb_dim // num_heads
    scaling = float(head_dim) ** -0.5
    q = q_proj(x)  # L, N, E
    k = k_proj(x)  # L, N, E
    v = v_proj(x)  # L, N, E

    v = v.reshape((-1, batch_size * num_heads, head_dim)).transpose((1, 0, 2))
    k = k.reshape((-1, batch_size * num_heads, head_dim)).transpose((1, 0, 2))
    q = q.reshape((-1, batch_size * num_heads, head_dim)).transpose((1, 0, 2))

    q = q * scaling
    qk = paddle.matmul(q, k, transpose_y=True)
    if attn_mask is not None:
        if attn_mask.ndim == 2:
            attn_mask.unsqueeze_(0)
        assert attn_mask.shape[0] == 1 and attn_mask.shape[1] == max_len and attn_mask.shape[2] == max_len
        qk += attn_mask

    qk = F.softmax(qk, axis=-1)
    atten = paddle.bmm(qk, v)
    atten = atten.transpose((1, 0, 2))
    atten = atten.reshape((max_len, batch_size, emb_dim))
    atten = c_proj(atten)
    return atten


class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2D(inplanes, planes, 1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)

        self.conv2 = nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)

        self.avgpool = nn.AvgPool2D(stride) if stride > 1 else Identity()

        self.conv3 = nn.Conv2D(planes, planes * self.expansion, 1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(
                ("-1", nn.AvgPool2D(stride)),
                ("0", nn.Conv2D(inplanes, planes * self.expansion, 1, stride=1, bias_attr=False)),
                ("1", nn.BatchNorm2D(planes * self.expansion)),
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Layer):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()

        self.positional_embedding = nn.Embedding(spacial_dim**2 + 1, embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias_attr=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias_attr=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias_attr=True)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim, bias_attr=True)
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

    def forward(self, x):

        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3])).transpose((2, 0, 1))  # NCHW -> (HW)NC
        x = paddle.concat([x.mean(axis=0, keepdim=True), x], axis=0)
        x = x + paddle.unsqueeze(self.positional_embedding.weight, 1)
        out = multi_head_attention_forward(x, self.num_heads, self.q_proj, self.k_proj, self.v_proj, self.c_proj)

        return out[0]


class CLIPPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained CLIP models. It provides CLIP related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    config_class = CLIPConfig
    base_model_prefix = "clip"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    @classmethod
    def _get_name_mappings(cls, config: CLIPConfig) -> List[StateDictNameMapping]:
        mappings: List[StateDictNameMapping] = []

        model_type = config.get("model_type", "clip")

        num_layer_key = "num_hidden_layers"
        num_text_layer = 0
        num_vision_layer = 0

        if model_type in ["clip", "clip_text_model"]:
            text_config = config.get("text_config")
            if text_config:
                num_text_layer = text_config.get(num_layer_key, 0)
            else:
                num_text_layer = config.get(num_layer_key, 0)

        if model_type in ["clip", "clip_vision_model"]:
            vision_config = config.get("vision_config")
            if vision_config:
                num_vision_layer = vision_config.get(num_layer_key, 0)
            else:
                num_vision_layer = config.get(num_layer_key, 0)

        has_text_layer = num_text_layer > 0
        has_text_projection_layer = has_text_layer and (
            "CLIPModel" in (config.architectures or [])
            or "CLIPTextModelWithProjection" in (config.architectures or [])
            or cls.__name__ in ["CLIPModel", "CLIPTextModelWithProjection"]
        )

        has_vision_layer = num_vision_layer > 0
        has_vision_projection_layer = has_vision_layer and (
            "CLIPModel" in (config.architectures or [])
            or "CLIPVisionModelWithProjection" in (config.architectures or [])
            or cls.__name__ in ["CLIPModel", "CLIPVisionModelWithProjection"]
        )

        if model_type == "clip":
            hard_mappings = [["logit_scale", "logit_scale"]]
        else:
            hard_mappings = []

        # text model
        if has_text_layer:
            text_model_layer_mappings = [
                ["text_model.embeddings.token_embedding.weight", "text_model.token_embedding.weight"],
                ["text_model.embeddings.position_embedding.weight", "text_model.positional_embedding.weight"],
                ["text_model.final_layer_norm.weight", "text_model.ln_final.weight"],
                ["text_model.final_layer_norm.bias", "text_model.ln_final.bias"],
            ]

            if has_text_projection_layer:
                text_model_layer_mappings.extend([["text_projection.weight", "text_projection", "transpose"]])

            hard_mappings.extend(text_model_layer_mappings)

            for layer_index in range(num_text_layer):
                text_model_layer_mappings = [
                    # qkv out
                    [
                        f"text_model.encoder.layers.{layer_index}.self_attn.q_proj.weight",
                        f"text_model.transformer.layers.{layer_index}.self_attn.q_proj.weight",
                        "transpose",
                    ],
                    [
                        f"text_model.encoder.layers.{layer_index}.self_attn.q_proj.bias",
                        f"text_model.transformer.layers.{layer_index}.self_attn.q_proj.bias",
                    ],
                    [
                        f"text_model.encoder.layers.{layer_index}.self_attn.k_proj.weight",
                        f"text_model.transformer.layers.{layer_index}.self_attn.k_proj.weight",
                        "transpose",
                    ],
                    [
                        f"text_model.encoder.layers.{layer_index}.self_attn.k_proj.bias",
                        f"text_model.transformer.layers.{layer_index}.self_attn.k_proj.bias",
                    ],
                    [
                        f"text_model.encoder.layers.{layer_index}.self_attn.v_proj.weight",
                        f"text_model.transformer.layers.{layer_index}.self_attn.v_proj.weight",
                        "transpose",
                    ],
                    [
                        f"text_model.encoder.layers.{layer_index}.self_attn.v_proj.bias",
                        f"text_model.transformer.layers.{layer_index}.self_attn.v_proj.bias",
                    ],
                    [
                        f"text_model.encoder.layers.{layer_index}.self_attn.out_proj.weight",
                        f"text_model.transformer.layers.{layer_index}.self_attn.out_proj.weight",
                        "transpose",
                    ],
                    [
                        f"text_model.encoder.layers.{layer_index}.self_attn.out_proj.bias",
                        f"text_model.transformer.layers.{layer_index}.self_attn.out_proj.bias",
                    ],
                    # fc1
                    [
                        f"text_model.encoder.layers.{layer_index}.mlp.fc1.weight",
                        f"text_model.transformer.layers.{layer_index}.linear1.weight",
                        "transpose",
                    ],
                    [
                        f"text_model.encoder.layers.{layer_index}.mlp.fc1.bias",
                        f"text_model.transformer.layers.{layer_index}.linear1.bias",
                    ],
                    [
                        f"text_model.encoder.layers.{layer_index}.layer_norm1.weight",
                        f"text_model.transformer.layers.{layer_index}.norm1.weight",
                    ],
                    [
                        f"text_model.encoder.layers.{layer_index}.layer_norm1.bias",
                        f"text_model.transformer.layers.{layer_index}.norm1.bias",
                    ],
                    # fc2
                    [
                        f"text_model.encoder.layers.{layer_index}.mlp.fc2.weight",
                        f"text_model.transformer.layers.{layer_index}.linear2.weight",
                        "transpose",
                    ],
                    [
                        f"text_model.encoder.layers.{layer_index}.mlp.fc2.bias",
                        f"text_model.transformer.layers.{layer_index}.linear2.bias",
                    ],
                    [
                        f"text_model.encoder.layers.{layer_index}.layer_norm2.weight",
                        f"text_model.transformer.layers.{layer_index}.norm2.weight",
                    ],
                    [
                        f"text_model.encoder.layers.{layer_index}.layer_norm2.bias",
                        f"text_model.transformer.layers.{layer_index}.norm2.bias",
                    ],
                ]
                hard_mappings.extend(text_model_layer_mappings)

        # vision model
        if has_vision_layer:
            vision_model_layer_mappings = [
                ["vision_model.embeddings.class_embedding", "vision_model.class_embedding"],
                ["vision_model.embeddings.patch_embedding.weight", "vision_model.conv1.weight"],
                ["vision_model.embeddings.position_embedding.weight", "vision_model.positional_embedding.weight"],
                ["vision_model.pre_layrnorm.weight", "vision_model.ln_pre.weight"],
                ["vision_model.pre_layrnorm.bias", "vision_model.ln_pre.bias"],
                ["vision_model.post_layernorm.weight", "vision_model.ln_post.weight"],
                ["vision_model.post_layernorm.bias", "vision_model.ln_post.bias"],
            ]

            if has_vision_projection_layer:
                vision_model_layer_mappings.extend([["visual_projection.weight", "vision_projection", "transpose"]])

            hard_mappings.extend(vision_model_layer_mappings)
            for layer_index in range(num_vision_layer):
                vision_model_layer_mappings = [
                    # qkv out
                    [
                        f"vision_model.encoder.layers.{layer_index}.self_attn.q_proj.weight",
                        f"vision_model.transformer.layers.{layer_index}.self_attn.q_proj.weight",
                        "transpose",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.self_attn.q_proj.bias",
                        f"vision_model.transformer.layers.{layer_index}.self_attn.q_proj.bias",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.self_attn.k_proj.weight",
                        f"vision_model.transformer.layers.{layer_index}.self_attn.k_proj.weight",
                        "transpose",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.self_attn.k_proj.bias",
                        f"vision_model.transformer.layers.{layer_index}.self_attn.k_proj.bias",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.self_attn.v_proj.weight",
                        f"vision_model.transformer.layers.{layer_index}.self_attn.v_proj.weight",
                        "transpose",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.self_attn.v_proj.bias",
                        f"vision_model.transformer.layers.{layer_index}.self_attn.v_proj.bias",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.self_attn.out_proj.weight",
                        f"vision_model.transformer.layers.{layer_index}.self_attn.out_proj.weight",
                        "transpose",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.self_attn.out_proj.bias",
                        f"vision_model.transformer.layers.{layer_index}.self_attn.out_proj.bias",
                    ],
                    # fc1
                    [
                        f"vision_model.encoder.layers.{layer_index}.mlp.fc1.weight",
                        f"vision_model.transformer.layers.{layer_index}.linear1.weight",
                        "transpose",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.mlp.fc1.bias",
                        f"vision_model.transformer.layers.{layer_index}.linear1.bias",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.layer_norm1.weight",
                        f"vision_model.transformer.layers.{layer_index}.norm1.weight",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.layer_norm1.bias",
                        f"vision_model.transformer.layers.{layer_index}.norm1.bias",
                    ],
                    # fc2
                    [
                        f"vision_model.encoder.layers.{layer_index}.mlp.fc2.weight",
                        f"vision_model.transformer.layers.{layer_index}.linear2.weight",
                        "transpose",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.mlp.fc2.bias",
                        f"vision_model.transformer.layers.{layer_index}.linear2.bias",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.layer_norm2.weight",
                        f"vision_model.transformer.layers.{layer_index}.norm2.weight",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.layer_norm2.bias",
                        f"vision_model.transformer.layers.{layer_index}.norm2.bias",
                    ],
                ]
                hard_mappings.extend(vision_model_layer_mappings)

        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(hard_mappings)]
        return mappings

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, nn.TransformerEncoder):
            module.enable_recompute = value

    def gradient_checkpointing_enable(self):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        self.apply(partial(self._set_gradient_checkpointing, value=True))

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if self.supports_gradient_checkpointing:
            self.apply(partial(self._set_gradient_checkpointing, value=False))

    def _init_weights(self, layer):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(layer, CLIPVisionTransformer):
            vision_embed_dim = layer.config.hidden_size
            vision_layers = layer.config.num_hidden_layers
            initializer_range = layer.config.initializer_range

            # vision embedding
            normal_(layer.class_embedding, std=vision_embed_dim**-0.5 * factor)
            normal_(layer.conv1.weight, std=initializer_range * factor)
            normal_(layer.positional_embedding.weight, std=initializer_range * factor)

            # init CLIPAttention + CLIPMLP
            for sub_layer in layer.sublayers():
                if isinstance(sub_layer, nn.TransformerEncoderLayer):
                    # self_attn
                    in_proj_std = (sub_layer.self_attn.embed_dim**-0.5) * ((2 * vision_layers) ** -0.5) * factor
                    out_proj_std = (sub_layer.self_attn.embed_dim**-0.5) * factor
                    normal_(sub_layer.self_attn.q_proj.weight, std=in_proj_std)
                    normal_(sub_layer.self_attn.k_proj.weight, std=in_proj_std)
                    normal_(sub_layer.self_attn.v_proj.weight, std=in_proj_std)
                    normal_(sub_layer.self_attn.out_proj.weight, std=out_proj_std)
                    # ffn
                    in_proj_std = (sub_layer._config["d_model"] ** -0.5) * ((2 * vision_layers) ** -0.5) * factor
                    fc_std = (2 * sub_layer._config["d_model"]) ** -0.5 * factor
                    normal_(sub_layer.linear1.weight, std=fc_std)
                    normal_(sub_layer.linear2.weight, std=in_proj_std)

        elif isinstance(layer, CLIPTextTransformer):
            text_layers = layer.config.num_hidden_layers
            initializer_range = layer.config.initializer_range

            # text embedding
            normal_(layer.token_embedding.weight, std=factor * 0.02)
            normal_(layer.positional_embedding.weight, std=factor * 0.02)

            # init CLIPAttention + CLIPMLP
            for sub_layer in layer.sublayers():
                if isinstance(sub_layer, nn.TransformerEncoderLayer):
                    # self_attn
                    in_proj_std = (sub_layer.self_attn.embed_dim**-0.5) * ((2 * text_layers) ** -0.5) * factor
                    out_proj_std = (sub_layer.self_attn.embed_dim**-0.5) * factor
                    normal_(sub_layer.self_attn.q_proj.weight, std=in_proj_std)
                    normal_(sub_layer.self_attn.k_proj.weight, std=in_proj_std)
                    normal_(sub_layer.self_attn.v_proj.weight, std=in_proj_std)
                    normal_(sub_layer.self_attn.out_proj.weight, std=out_proj_std)
                    # ffn
                    in_proj_std = (sub_layer._config["d_model"] ** -0.5) * ((2 * text_layers) ** -0.5) * factor
                    fc_std = (2 * sub_layer._config["d_model"]) ** -0.5 * factor
                    normal_(sub_layer.linear1.weight, std=fc_std)
                    normal_(sub_layer.linear2.weight, std=in_proj_std)

        elif isinstance(layer, ModifiedResNet):
            if layer.attnpool is not None:
                std = layer.output_dim**-0.5
                normal_(layer.attnpool.q_proj.weight, std=std)
                normal_(layer.attnpool.k_proj.weight, std=std)
                normal_(layer.attnpool.v_proj.weight, std=std)
                normal_(layer.attnpool.c_proj.weight, std=std)

            for resnet_block in [layer.layer1, layer.layer2, layer.layer3, layer.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        zeros_(param)

        elif isinstance(layer, CLIPModel):
            normal_(layer.text_projection, std=layer.text_embed_dim**-0.5 * self.config.initializer_factor)
            if hasattr(layer, "vision_projection"):
                normal_(layer.vision_projection, std=layer.vision_embed_dim**-0.5 * self.config.initializer_factor)
        elif isinstance(layer, CLIPVisionModelWithProjection):
            if hasattr(layer, "vision_projection"):
                normal_(layer.vision_projection, std=self.config.hidden_size**-0.5 * self.config.initializer_factor)
        elif isinstance(layer, CLIPTextModelWithProjection):
            normal_(layer.text_projection, std=self.config.hidden_size**-0.5 * self.config.initializer_factor)

        if isinstance(layer, nn.LayerNorm):
            zeros_(layer.bias)
            ones_(layer.weight)

        if isinstance(layer, nn.Linear) and layer.bias is not None:
            zeros_(layer.bias)


class CLIPTextTransformer(nn.Layer):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            normalize_before=True,
            dropout=0.0,
            activation=config.hidden_act,
            attn_dropout=config.attention_dropout,
            act_dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_hidden_layers)
        self.ln_final = nn.LayerNorm(embed_dim)

        # For `pooled_output` computation
        self.eos_token_id = config.eos_token_id

        self.register_buffer(
            "causal_mask",
            paddle.triu(
                paddle.ones((1, 1, config.max_position_embeddings, config.max_position_embeddings)) * NEG_INF,
                diagonal=1,
            ),
            persistable=False,
        )
        self.register_buffer(
            "position_ids",
            paddle.arange(config.max_position_embeddings, dtype="int64").reshape((1, -1)),
            persistable=False,
        )

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Args:
            input_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
                Indices can be obtained using [`CLIPTokenizer`].
            attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            position_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.max_position_embeddings - 1]`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`BaseModelOutputWithPooling`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bs, seqlen = input_ids.shape
        if position_ids is None:
            position_ids = self.position_ids[:, :seqlen].cast("int64")

        embedding_output = self.token_embedding(input_ids) + self.positional_embedding(
            position_ids
        )  # [batch_size, n_ctx, d_model]

        causal_mask = self.causal_mask[:, :, :seqlen, :seqlen]
        if attention_mask is not None:
            assert attention_mask.ndim == 2
            expanded_mask = attention_mask[:, None, None, :].expand([bs, 1, seqlen, -1]).cast(causal_mask.dtype)
            inverted_mask = (1.0 - expanded_mask) * NEG_INF
            attention_mask = inverted_mask + causal_mask
        else:
            attention_mask = causal_mask
        attention_mask.stop_gradient = True

        encoder_outputs = self.transformer(
            embedding_output,
            src_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if isinstance(encoder_outputs, type(embedding_output)):
            last_hidden_state = encoder_outputs
        else:
            last_hidden_state = encoder_outputs[0]

        last_hidden_state = self.ln_final(last_hidden_state)

        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to paddle.int32 for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state.gather_nd(
                paddle.stack(
                    [paddle.arange(last_hidden_state.shape[0], dtype="int32"), input_ids.argmax(-1, dtype="int32")],
                    axis=-1,
                )
            )
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
            pooled_output = last_hidden_state.gather_nd(
                paddle.stack(
                    [
                        paddle.arange(last_hidden_state.shape[0], dtype="int32"),
                        (input_ids == self.eos_token_id).cast("int32").argmax(axis=-1, dtype="int32"),
                    ],
                    axis=-1,
                )
            )

        if isinstance(encoder_outputs, type(embedding_output)):
            return (last_hidden_state, pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPTextModel(CLIPPretrainedModel):
    r"""
    The text model from CLIP without any head or projection on top.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`CLIPTextConfig`):
            An instance of CLIPTextConfig used to construct CLIPTextModel.
    """

    config_class = CLIPTextConfig

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = CLIPTextTransformer(config)

    def get_input_embeddings(self) -> nn.Layer:
        return self.text_model.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.token_embedding = value

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Args:
            input_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
                Indices can be obtained using [`CLIPTokenizer`].
            attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            position_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.max_position_embeddings - 1]`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`BaseModelOutputWithPooling`] instead of a plain tuple.

        Returns:
            An instance of :class:`BaseModelOutputWithPooling` if `return_dict=True`. Otherwise it returns a tuple of tensors
            corresponding to ordered and not None (depending on the input arguments) fields of :class:`BaseModelOutputWithPooling`.

        Examples:

        ```python
        >>> from paddlenlp.transformers import CLIPTokenizer, CLIPTextModel

        >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pd")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CLIPVisionTransformer(nn.Layer):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.input_resolution = config.image_size
        self.class_embedding = self.create_parameter(
            (embed_dim,),
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Assign(paddle.randn((embed_dim,))),
        )
        self.conv1 = nn.Conv2D(
            in_channels=config.num_channels,
            out_channels=embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias_attr=False,
        )
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.positional_embedding = nn.Embedding(self.num_positions, embed_dim)

        self.ln_pre = nn.LayerNorm(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            normalize_before=True,
            dropout=0.0,
            activation=config.hidden_act,
            attn_dropout=config.attention_dropout,
            act_dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_hidden_layers)
        self.ln_post = nn.LayerNorm(embed_dim)
        self.register_buffer(
            "position_ids",
            paddle.arange(self.num_positions).reshape((1, -1)),
            persistable=False,
        )

    def forward(
        self,
        pixel_values: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Args:
            pixel_values (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
                [`CLIPFeatureExtractor`]. See [`CLIPFeatureExtractor.__call__`] for details.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`BaseModelOutputWithPooling`] instead of a plain tuple.

        Returns:
            An instance of :class:`BaseModelOutputWithPooling` if `return_dict=True`. Otherwise it returns a tuple of tensors
            corresponding to ordered and not None (depending on the input arguments) fields of :class:`BaseModelOutputWithPooling`.

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        target_dtype = self.conv1.weight.dtype
        pixel_values = self.conv1(pixel_values.cast(target_dtype))

        pixel_values = pixel_values.reshape((pixel_values.shape[0], pixel_values.shape[1], -1))
        pixel_values = pixel_values.transpose((0, 2, 1))
        embedding_output = paddle.concat(
            [self.class_embedding.unsqueeze([0, 1]).expand([pixel_values.shape[0], -1, -1]), pixel_values], axis=1
        )
        hidden_states = embedding_output + self.positional_embedding.weight
        hidden_states = self.ln_pre(hidden_states)

        encoder_outputs = self.transformer(
            hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if isinstance(encoder_outputs, type(embedding_output)):
            last_hidden_state = encoder_outputs
        else:
            last_hidden_state = encoder_outputs[0]

        pooled_output = self.ln_post(last_hidden_state[:, 0])

        if isinstance(encoder_outputs, type(embedding_output)):
            return (last_hidden_state, pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward_pre(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape([x.shape[0], x.shape[1], -1])  # shape = [*, width, grid ** 2]
        x = x.transpose((0, 2, 1))  # shape = [*, grid ** 2, width]
        # t = self.class_embedding.weight + paddle.zeros([x.shape[0], 1, x.shape[-1]], dtype=x.dtype)
        t = self.class_embedding.unsqueeze([0, 1]).expand([x.shape[0], -1, -1]) + paddle.zeros(
            [x.shape[0], 1, x.shape[-1]], dtype=x.dtype
        )
        x = paddle.concat([t, x], axis=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.weight
        x = self.ln_pre(x)
        return x

    def forward_post(self, x):
        x = self.ln_post(x)
        return x


class CLIPVisionModel(CLIPPretrainedModel):
    r"""
    The vision model from CLIP without any head or projection on top.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`CLIPVisionConfig`):
            An instance of CLIPVisionConfig used to construct CLIPVisionModel.
    """
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        if isinstance(config.num_hidden_layers, (tuple, list)):
            raise NotImplementedError("We only support VIT CLIP Vision Transformer!")

        self.vision_model = CLIPVisionTransformer(config)

    def get_input_embeddings(self) -> nn.Layer:
        return self.vision_model.conv1

    def forward(
        self,
        pixel_values: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Args:
            pixel_values (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
                [`CLIPFeatureExtractor`]. See [`CLIPFeatureExtractor.__call__`] for details.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`BaseModelOutputWithPooling`] instead of a plain tuple.

        Returns:
            An instance of :class:`BaseModelOutputWithPooling` if `return_dict=True`. Otherwise it returns a tuple of tensors
            corresponding to ordered and not None (depending on the input arguments) fields of :class:`BaseModelOutputWithPooling`.

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from paddlenlp.transformers import CLIPProcessor, CLIPVisionModel

        >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pd")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CLIPModel(CLIPPretrainedModel):
    r"""
    The bare CLIP Model outputting logits_per_image and logits_per_text.
    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`CLIPConfig`):
            An instance of CLIPConfig used to construct CLIPModel.
    """
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config)

        if isinstance(vision_config.num_hidden_layers, (tuple, list)):
            if vision_config.num_attention_heads is None:
                vision_heads = vision_config.hidden_size * 32 // 64
            else:
                vision_heads = vision_config.num_attention_heads
            self.vision_model = ModifiedResNet(
                layers=vision_config.num_hidden_layers,
                output_dim=self.projection_dim,
                heads=vision_heads,
                input_resolution=vision_config.image_size,
                width=vision_config.hidden_size,
            )
        else:
            self.vision_model = CLIPVisionTransformer(vision_config)
            self.vision_projection = paddle.create_parameter(
                (self.vision_embed_dim, self.projection_dim), paddle.get_default_dtype()
            )
        self.text_projection = paddle.create_parameter(
            (self.text_embed_dim, self.projection_dim), paddle.get_default_dtype()
        )

        self.logit_scale = paddle.create_parameter(
            (1,),
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(config.logit_scale_init_value),
        )

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
        Args:
            input_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
                Indices can be obtained using [`CLIPTokenizer`].
            attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            position_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.max_position_embeddings - 1]`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`BaseModelOutputWithPooling`] instead of a plain tuple.

        Returns:
            text_features (`paddle.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTextModel`].

        Examples:

        ```python
        >>> from paddlenlp.transformers import CLIPTokenizer, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pd")
        >>> text_features = model.get_text_features(**inputs)
        ```
        """
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
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

        pooled_output = text_outputs[1]
        text_features = paddle.matmul(pooled_output, self.text_projection)

        return text_features

    def get_image_features(
        self,
        pixel_values: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> paddle.Tensor:
        r"""
        Args:
            pixel_values (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
                [`CLIPFeatureExtractor`]. See [`CLIPFeatureExtractor.__call__`] for details.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`BaseModelOutputWithPooling`] instead of a plain tuple.

        Returns:
            image_features (`paddle.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from paddlenlp.transformers import CLIPProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pd")

        >>> image_features = model.get_image_features(**inputs)
        ```
        """
        if isinstance(self.vision_model, ModifiedResNet):
            return self.vision_model(pixel_values)
        else:
            # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            pooled_output = vision_outputs[1]  # pooled_output
            image_features = paddle.matmul(pooled_output, self.vision_projection)

            return image_features

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPOutput]:
        r"""
        The CLIPModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
                Its data type should be `int64` and it has a shape of [text_batch_size, sequence_length].
            pixel_values (Tensor):
                Pixel values. Padding will be ignored by default should you provide it.
                Its data type should be `float32` and it has a shape of [image_batch_size, num_channels, height, width].
            position_ids(Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings (CLIPTextTransformer). Selected in
                the range ``[0, max_text_length - 1]``.
                Shape as `(batch_size, num_tokens)` and dtype as int64. Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in multi-head attention (CLIPTextTransformer) to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `0.0` values and the others have `1.0` values.
                It is a tensor with shape `[batch_size, sequence_length`.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`CLIPOutput` object. If `False`, the output
                will be a tuple of tensors. Defaults to `True`.

        Returns:
            An instance of :class:`CLIPOutput` if `return_dict=True`. Otherwise it returns a tuple of tensors
            corresponding to ordered and not None (depending on the input arguments) fields of :class:`CLIPOutput`.

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> import paddle.nn.functional as F
        >>> from paddlenlp.transformers import CLIPProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> model.eval()
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pd", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = F.softmax(logits_per_image, axis=1)  # we can take the softmax to get the label probabilities
        ```
        """
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if isinstance(self.vision_model, ModifiedResNet):
            vision_outputs = None
            image_embeds = self.vision_model(pixel_values)
        else:
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            image_embeds = vision_outputs[1]
            image_embeds = paddle.matmul(image_embeds, self.vision_projection)

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_embeds = text_outputs[1]
        text_embeds = paddle.matmul(text_embeds, self.text_projection)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(axis=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(axis=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = paddle.matmul(text_embeds, image_embeds, transpose_y=True) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class CLIPTextModelWithProjection(CLIPPretrainedModel):
    r"""
    CLIP Text Model with a projection layer on top (a linear layer on top of the pooled output).

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`CLIPTextConfig`):
            An instance of CLIPTextConfig used to construct CLIPTextModelWithProjection.
    """
    config_class = CLIPTextConfig

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)

        self.text_model = CLIPTextTransformer(config)

        self.text_projection = paddle.create_parameter(
            (config.hidden_size, config.projection_dim), paddle.get_default_dtype()
        )

    def get_input_embeddings(self) -> nn.Layer:
        return self.text_model.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.token_embedding = value

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPTextModelOutput]:
        r"""
        Args:
            input_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
                Indices can be obtained using [`CLIPTokenizer`].
            attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            position_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.max_position_embeddings - 1]`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`CLIPTextModelOutput`] instead of a plain tuple.
                If `False`, the output will be a tuple of tensors. Defaults to `None`.

        Returns:
            An instance of :class:`CLIPTextModelOutput` if `return_dict=True`. Otherwise it returns a tuple of tensors
            corresponding to ordered and not None (depending on the input arguments) fields of :class:`CLIPTextModelOutput`.

        Examples:

        ```python
        >>> from paddlenlp.transformers import CLIPTokenizer, CLIPTextModelWithProjection

        >>> model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pd")

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

        pooled_output = text_outputs[1]

        text_embeds = paddle.matmul(pooled_output, self.text_projection)

        if not return_dict:
            outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return CLIPTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=text_outputs.last_hidden_state,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )


class CLIPVisionModelWithProjection(CLIPPretrainedModel):
    r"""
    CLIP Vision Model with a projection layer on top (a linear layer on top of the pooled output).

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`CLIPVisionConfig`):
            An instance of CLIPVisionConfig used to construct CLIPVisionModelWithProjection.
    """
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)

        # support resnet vision model
        if isinstance(config.num_hidden_layers, (tuple, list)):
            if config.num_attention_heads is None:
                vision_heads = config.hidden_size * 32 // 64
            else:
                vision_heads = config.num_attention_heads
            self.vision_model = ModifiedResNet(
                layers=config.num_hidden_layers,
                output_dim=config.projection_dim,
                heads=vision_heads,
                input_resolution=config.image_size,
                width=config.hidden_size,
            )
        else:
            self.vision_model = CLIPVisionTransformer(config)
            self.vision_projection = paddle.create_parameter(
                (config.hidden_size, config.projection_dim), paddle.get_default_dtype()
            )

    def get_input_embeddings(self) -> nn.Layer:
        if isinstance(self.vision_model, CLIPVisionTransformer):
            return self.vision_model.conv1
        else:
            return None

    def forward(
        self,
        pixel_values: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPVisionModelOutput]:
        r"""
        Args:
            pixel_values (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
                [`CLIPFeatureExtractor`]. See [`CLIPFeatureExtractor.__call__`] for details.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`CLIPVisionModelOutput`] instead of a plain tuple.

        Returns:
            An instance of :class:`CLIPVisionModelOutput` if `return_dict=True`. Otherwise it returns a tuple of tensors
            corresponding to ordered and not None (depending on the input arguments) fields of :class:`CLIPVisionModelOutput`.

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from paddlenlp.transformers import CLIPProcessor, CLIPVisionModelWithProjection

        >>> model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pd")

        >>> outputs = model(**inputs)
        >>> image_embeds = outputs.image_embeds
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if isinstance(self.vision_model, ModifiedResNet):
            image_embeds = self.vision_model(pixel_values)
            if not return_dict:
                return (image_embeds,)
            else:
                return CLIPVisionModelOutput(image_embeds=image_embeds)
        else:
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            pooled_output = vision_outputs[1]  # pooled_output

            image_embeds = paddle.matmul(pooled_output, self.vision_projection)

            if not return_dict:
                outputs = (image_embeds, vision_outputs[0]) + vision_outputs[2:]
                return tuple(output for output in outputs if output is not None)

            return CLIPVisionModelOutput(
                image_embeds=image_embeds,
                last_hidden_state=vision_outputs.last_hidden_state,
                hidden_states=vision_outputs.hidden_states,
                attentions=vision_outputs.attentions,
            )
