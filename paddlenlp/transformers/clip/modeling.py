# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The Open AI Team Authors and The HuggingFace Inc. team.
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

from typing import Any, Tuple, Optional, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from dataclasses import dataclass
from ..model_outputs import BaseModelOutputWithPoolingAndCrossAttentions, ModelOutput
from .. import PretrainedModel, register_base_model
from ..stable_diffusion_utils import (
    StableDiffusionMixin,
    AutoencoderKL,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
)
from ..guided_diffusion_utils import (
    DiscoDiffusionMixin,
    create_gaussian_diffusion,
    create_unet_model,
    create_secondary_model,
)

__all__ = [
    "VisionTransformer",
    "TextTransformer",
    "CLIPTextModel",
    "CLIPVisionModel",
    "CLIPPretrainedModel",
    "CLIPModel",
    "CLIPForImageGeneration",
    "ModifiedResNet",
]


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
            The text embeddings obtained by applying the projection layer to the pooled output of [`TextTransformer`].
        image_embeds(`paddle.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`VisionTransformer`].
        text_model_output(:class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions`):
            The output of the [`TextTransformer`].
        vision_model_output(:class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions`):
            The output of the [`VisionTransformer`].
    """

    loss: Optional[paddle.Tensor] = None
    logits_per_image: paddle.Tensor = None
    logits_per_text: paddle.Tensor = None
    text_embeds: paddle.Tensor = None
    image_embeds: paddle.Tensor = None
    text_model_output: BaseModelOutputWithPoolingAndCrossAttentions = None
    vision_model_output: BaseModelOutputWithPoolingAndCrossAttentions = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits):
    return F.cross_entropy(logits, paddle.arange(len(logits)))


def clip_loss(similarity):
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


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

    pretrained_init_configuration = {
        "openai/clip-vit-base-patch32": {
            # vision
            "image_resolution": 224,
            "vision_layers": 12,
            "vision_heads": 12,
            "vision_mlp_ratio": 4,
            "vision_embed_dim": 768,
            "vision_patch_size": 32,
            "vision_hidden_act": "quick_gelu",
            # text
            "max_text_length": 77,
            "vocab_size": 49408,
            "text_embed_dim": 512,
            "text_heads": 8,
            "text_layers": 12,
            "text_hidden_act": "quick_gelu",
            # others
            "projection_dim": 512,
            "initializer_range": 0.02,
            "logit_scale_init_value": 2.6592,
        },
        "openai/clip-rn50": {
            # vision
            "image_resolution": 224,
            "vision_layers": [3, 4, 6, 3],
            "vision_heads": 32,
            "vision_mlp_ratio": None,  # do not use
            "vision_embed_dim": 64,  # vision width
            "vision_patch_size": None,  # do not use
            "vision_hidden_act": None,  # do not use
            # text
            "max_text_length": 77,
            "vocab_size": 49408,
            "text_embed_dim": 512,
            "text_heads": 8,
            "text_layers": 12,
            "text_hidden_act": "quick_gelu",
            # others
            "projection_dim": 1024,
            "initializer_range": 0.02,
            "logit_scale_init_value": 2.6592,
        },
        "openai/clip-rn101": {
            # vision
            "image_resolution": 224,
            "vision_layers": [3, 4, 23, 3],
            "vision_heads": 32,
            "vision_mlp_ratio": None,  # do not use
            "vision_embed_dim": 64,  # vision width
            "vision_patch_size": None,  # do not use
            "vision_hidden_act": None,  # do not use
            # text
            "max_text_length": 77,
            "vocab_size": 49408,
            "text_embed_dim": 512,
            "text_heads": 8,
            "text_layers": 12,
            "text_hidden_act": "quick_gelu",
            # others
            "projection_dim": 512,
            "initializer_range": 0.02,
            "logit_scale_init_value": 2.6592,
        },
        "openai/clip-vit-large-patch14": {
            # vision
            "image_resolution": 224,
            "vision_layers": 24,
            "vision_heads": 16,
            "vision_mlp_ratio": 4,
            "vision_embed_dim": 1024,
            "vision_patch_size": 14,
            "vision_hidden_act": "quick_gelu",
            # text
            "max_text_length": 77,
            "vocab_size": 49408,
            "text_embed_dim": 768,
            "text_heads": 12,
            "text_layers": 12,
            "text_hidden_act": "quick_gelu",
            # others
            "projection_dim": 768,
            "initializer_range": 0.02,
            "logit_scale_init_value": 2.6592,
        },
    }
    pretrained_resource_files_map = {
        "model_state": {
            "openai/clip-vit-base-patch32": "http://bj.bcebos.com/paddlenlp/models/community/openai/clip-vit-base-patch32/model_state.pdparams",
            "openai/clip-rn50": "http://bj.bcebos.com/paddlenlp/models/community/openai/clip-rn50/model_state.pdparams",
            "openai/clip-rn101": "http://bj.bcebos.com/paddlenlp/models/community/openai/clip-rn101/model_state.pdparams",
            "openai/clip-vit-large-patch14": "http://bj.bcebos.com/paddlenlp/models/community/openai/clip-vit-large-patch14/model_state.pdparams",
        }
    }
    base_model_prefix = "clip"

    def _init_weights(self, layer):
        """Initialize the weights"""
        initializer_range = (
            self.initializer_range if hasattr(self, "initializer_range") else self.clip.config["initializer_range"]
        )
        factor = (
            self.initializer_factor if hasattr(self, "initializer_factor") else self.clip.config["initializer_factor"]
        )

        if isinstance(layer, VisionTransformer):
            vision_embed_dim = (
                self.vision_embed_dim if hasattr(self, "vision_embed_dim") else self.clip.config["vision_embed_dim"]
            )
            vision_layers = self.vision_layers if hasattr(self, "vision_layers") else self.clip.config["vision_layers"]
            # vision embedding
            layer.class_embedding.set_value(
                paddle.normal(
                    std=vision_embed_dim**-0.5 * factor,
                    shape=layer.class_embedding.shape,
                )
            )
            layer.conv1.weight.set_value(
                paddle.normal(
                    std=initializer_range * factor,
                    shape=layer.conv1.weight.shape,
                )
            )
            layer.positional_embedding.weight.set_value(
                paddle.normal(
                    std=initializer_range * factor,
                    shape=layer.positional_embedding.weight.shape,
                )
            )

        elif isinstance(layer, TextTransformer):
            text_embed_dim = (
                self.text_embed_dim if hasattr(self, "text_embed_dim") else self.clip.config["text_embed_dim"]
            )
            text_layers = self.text_layers if hasattr(self, "text_layers") else self.clip.config["text_layers"]
            # text embedding
            layer.token_embedding.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * 0.02,
                    shape=layer.token_embedding.weight.shape,
                )
            )
            layer.positional_embedding.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * 0.02,
                    shape=layer.positional_embedding.weight.shape,
                )
            )
        elif isinstance(layer, CLIPModel):
            vision_embed_dim = (
                self.vision_embed_dim if hasattr(self, "vision_embed_dim") else self.clip.config["vision_embed_dim"]
            )
            vision_layers = self.vision_layers if hasattr(self, "vision_layers") else self.clip.config["vision_layers"]
            text_embed_dim = (
                self.text_embed_dim if hasattr(self, "text_embed_dim") else self.clip.config["text_embed_dim"]
            )
            text_layers = self.text_layers if hasattr(self, "text_layers") else self.clip.config["text_layers"]
            layer.text_projection.set_value(
                paddle.normal(
                    std=text_embed_dim**-0.5 * factor,
                    shape=layer.text_projection.shape,
                )
            )
            if hasattr(layer, "vision_projection"):
                layer.vision_projection.set_value(
                    paddle.normal(
                        std=vision_embed_dim**-0.5 * factor,
                        shape=layer.vision_projection.shape,
                    )
                )
            for name, sub_layer in layer.named_sublayers():
                num_layers = vision_layers if "vision_model" in name else text_layers
                if isinstance(sub_layer, nn.TransformerEncoderLayer):
                    # self_attn
                    in_proj_std = (sub_layer.self_attn.embed_dim**-0.5) * ((2 * num_layers) ** -0.5) * factor
                    out_proj_std = (sub_layer.self_attn.embed_dim**-0.5) * factor
                    sub_layer.self_attn.q_proj.weight.set_value(
                        paddle.normal(
                            std=in_proj_std,
                            shape=sub_layer.self_attn.q_proj.weight.shape,
                        )
                    )
                    sub_layer.self_attn.k_proj.weight.set_value(
                        paddle.normal(
                            std=in_proj_std,
                            shape=sub_layer.self_attn.k_proj.weight.shape,
                        )
                    )
                    sub_layer.self_attn.v_proj.weight.set_value(
                        paddle.normal(
                            std=in_proj_std,
                            shape=sub_layer.self_attn.v_proj.weight.shape,
                        )
                    )
                    sub_layer.self_attn.out_proj.weight.set_value(
                        paddle.normal(
                            std=out_proj_std,
                            shape=sub_layer.self_attn.out_proj.weight.shape,
                        )
                    )
                    # ffn
                    in_proj_std = (sub_layer._config["d_model"] ** -0.5) * ((2 * num_layers) ** -0.5) * factor
                    fc_std = (2 * sub_layer._config["d_model"]) ** -0.5 * factor
                    sub_layer.linear1.weight.set_value(
                        paddle.normal(
                            std=fc_std,
                            shape=sub_layer.linear1.weight.shape,
                        )
                    )
                    sub_layer.linear2.weight.set_value(
                        paddle.normal(
                            std=in_proj_std,
                            shape=sub_layer.linear2.weight.shape,
                        )
                    )
        if isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.ones_like(layer.weight))
        if isinstance(layer, nn.Linear) and layer.bias is not None:
            layer.bias.set_value(paddle.zeros_like(layer.bias))


# set attr
def quick_gelu(x):
    return x * F.sigmoid(1.702 * x)


F.quick_gelu = quick_gelu

NEG_INF = -1e4  # float("-inf") -1e4 -1e9


class VisionTransformer(nn.Layer):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        activation: str,
        mlp_ratio: int,
        normalize_before: bool = True,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        # used patch_size x patch_size, stride patch_size to do linear projection
        self.conv1 = nn.Conv2D(
            in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias_attr=False
        )

        self.class_embedding = paddle.create_parameter((width,), paddle.get_default_dtype())

        self.positional_embedding = nn.Embedding((input_resolution // patch_size) ** 2 + 1, width)

        self.ln_pre = nn.LayerNorm(width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * mlp_ratio,
            normalize_before=normalize_before,
            dropout=0,
            activation=activation,
            attn_dropout=0,
            act_dropout=0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)

        self.ln_post = nn.LayerNorm(width)

    def forward(self, pixel_values, output_attentions=False, output_hidden_states=False, return_dict=False):
        pixel_values = self.conv1(pixel_values)
        pixel_values = pixel_values.reshape((pixel_values.shape[0], pixel_values.shape[1], -1))
        pixel_values = pixel_values.transpose((0, 2, 1))
        embedding_output = paddle.concat(
            [self.class_embedding.unsqueeze([0, 1]).expand([pixel_values.shape[0], -1, -1]), pixel_values], axis=1
        )
        embedding_output = embedding_output + self.positional_embedding.weight
        embedding_output = self.ln_pre(embedding_output)
        encoder_outputs = self.transformer(
            embedding_output,
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

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TextTransformer(nn.Layer):
    def __init__(
        self,
        context_length,
        transformer_width,
        transformer_heads,
        transformer_layers,
        vocab_size,
        activation="quick_gelu",
        normalize_before=True,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_width,
            nhead=transformer_heads,
            dim_feedforward=transformer_width * 4,
            normalize_before=normalize_before,
            dropout=0,
            activation=activation,
            attn_dropout=0,
            act_dropout=0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, transformer_layers)

        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Embedding(context_length, transformer_width)
        self.ln_final = nn.LayerNorm(transformer_width)

        self.register_buffer(
            "causal_mask",
            paddle.triu(paddle.ones((1, 1, context_length, context_length)) * NEG_INF, diagonal=1),
            persistable=False,
        )
        self.register_buffer("position_ids", paddle.arange(context_length).reshape((1, -1)), persistable=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        bs, seqlen = input_ids.shape[:2]
        if position_ids is None:
            position_ids = self.position_ids[:, :seqlen].astype("int64")

        causal_mask = self.causal_mask[:, :, :seqlen, :seqlen]
        if attention_mask is not None:
            assert attention_mask.ndim == 2
            expanded_mask = attention_mask[:, None, None, :].expand([bs, 1, seqlen, -1]).astype(causal_mask.dtype)
            inverted_mask = (1.0 - expanded_mask) * NEG_INF
            attention_mask = inverted_mask + causal_mask
        else:
            attention_mask = causal_mask
        attention_mask.stop_gradient = True

        embedding_output = self.token_embedding(input_ids) + self.positional_embedding(
            position_ids
        )  # [batch_size, n_ctx, d_model]

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
        pooled_output = last_hidden_state.gather_nd(paddle.stack([paddle.arange(bs), input_ids.argmax(-1)], axis=-1))

        if isinstance(encoder_outputs, type(embedding_output)):
            return (last_hidden_state, pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@register_base_model
class CLIPModel(CLIPPretrainedModel):
    r"""
    The bare CLIP Model outputting logits_per_image and logits_per_text.
    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        image_resolution (int, optional):
            The size (resolution) of each image.
            Defaults to `224`.
        vision_layers (int, optional):
            Number of hidden layers in the vision model.
            Defaults to `12`.
        vision_heads (int, optional):
            Number of attention heads for each attention layer in the vision attention.
            Defaults to `12`.
        vision_embed_dim (int, optional):
            Dimensionality of the embedding layer and encoder layers in vision model.
            Defaults to `768`.
        vision_patch_size(int, optional):
            The size (resolution) of each patch.
            Defaults to `32`.
        vision_mlp_ratio(int, optional):
            The ratio between dim_feedforward and vision_hidden_dim. `radio = dim_feedforward/vision_hidden_dim`
            Defaults to `4`.
        vision_hidden_act (str, optional):
            The non-linear activation function of the ffn layer in the vision model.
            ``"gelu"``, ``"relu"``, ``"quick_gelu"`` and any other paddle supported activation functions are supported.
            Defaults to `"quick_gelu"`.
        max_text_length (int, optional):
            The maximum value of the dimensionality of text position encoding, which dictates the maximum supported length of the text
            input sequence. Defaults to `64`.
        vocab_size (int, optional):
            Vocabulary size of `inputs_ids` in `CLIPModel`. Also is the vocab size of text token embedding matrix.
            Defaults to `49408`.
        text_embed_dim (int, optional):
            Dimensionality of the embedding layer and encoder layers in text model.
            Defaults to `768`.
        text_heads (int, optional):
            Number of attention heads for each attention layer in the text attention.
            Defaults to `8`.
        text_layers (int, optional):
            Number of hidden layers in the text model.
            Defaults to `12`.
        text_hidden_act (str, optional):
            The non-linear activation function of the ffn layer in the text model.
            ``"gelu"``, ``"relu"``, ``"quick_gelu"`` and any other paddle supported activation functions are supported.
            Defaults to `"quick_gelu"`.
        projection_dim (int, optional):
            Dimentionality of text and vision projection layers.
            Defaults to `512`.
        initializer_range (float, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Default to `0.02`.
        initializer_factor (float, optional):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing). Default to `1.`.
        logit_scale_init_value (float, optional):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLIP implementation.
            Default to `2.6592`.

    """

    def __init__(
        self,
        # vision
        image_resolution: int = 224,
        vision_layers: Union[Tuple[int, int, int, int], int] = 12,
        vision_heads: int = 12,
        vision_embed_dim: int = 768,
        vision_patch_size: int = 32,
        vision_mlp_ratio: int = 4,
        vision_hidden_act: str = "quick_gelu",
        # text
        max_text_length: int = 77,
        vocab_size: int = 49408,
        text_embed_dim: int = 512,
        text_heads: int = 8,
        text_layers: int = 12,
        text_hidden_act: str = "quick_gelu",
        # others
        projection_dim: int = 512,
        initializer_range: float = 0.02,
        initializer_factor: float = 1.0,
        logit_scale_init_value: float = 2.6592,
    ):
        super().__init__()
        self.initializer_factor = initializer_factor
        self.initializer_range = initializer_range
        self.logit_scale_init_value = logit_scale_init_value
        self.vision_embed_dim = vision_embed_dim
        self.text_embed_dim = text_embed_dim
        self.vision_layers = vision_layers
        self.text_layers = text_layers

        if isinstance(vision_layers, (tuple, list)):
            if vision_heads is None:
                vision_heads = vision_embed_dim * 32 // 64
            self.vision_model = ModifiedResNet(
                layers=vision_layers,
                output_dim=projection_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_embed_dim,
            )
        else:
            if vision_heads is None:
                vision_heads = vision_embed_dim // 64
            self.vision_model = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_embed_dim,
                layers=vision_layers,
                heads=vision_heads,
                activation=vision_hidden_act,
                mlp_ratio=vision_mlp_ratio,
                normalize_before=True,
            )
            self.vision_projection = paddle.create_parameter(
                (vision_embed_dim, projection_dim), paddle.get_default_dtype()
            )

        self.text_model = TextTransformer(
            context_length=max_text_length,
            transformer_width=text_embed_dim,
            transformer_heads=text_heads,
            transformer_layers=text_layers,
            vocab_size=vocab_size,
            activation=text_hidden_act,
            normalize_before=True,
        )

        self.text_projection = paddle.create_parameter((text_embed_dim, projection_dim), paddle.get_default_dtype())

        self.logit_scale = paddle.create_parameter(
            (1,), dtype=paddle.get_default_dtype(), default_initializer=nn.initializer.Constant(logit_scale_init_value)
        )
        self.apply(self._init_weights)

    def get_image_features(
        self, pixel_values=None, output_attentions=False, output_hidden_states=False, return_dict=False
    ):
        r"""
        Returns:
            image_features (`paddle.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`VisionTransformer`].

        Examples:
            .. code-block::

                import requests
                from PIL import Image
                from paddlenlp.transformers import CLIPProcessor, CLIPModel

                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

                url = "http://images.cocodataset.org/val2017/000000039769.jpg"
                image = Image.open(requests.get(url, stream=True).raw)
                inputs = processor(images=image, return_tensors="pd")
                image_features = model.get_image_features(**inputs)

        """
        if isinstance(self.vision_model, ModifiedResNet):
            return self.vision_model(pixel_values)
        else:
            vision_outputs = self.vision_model(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            pooled_output = vision_outputs[1]
            image_features = paddle.matmul(pooled_output, self.vision_projection)
            return image_features

    def get_text_features(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        r"""
        Returns:
            text_features (`paddle.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`TextTransformer`].

        Example:
            .. code-block::

                from paddlenlp.transformers import CLIPModel, CLIPTokenizer

                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

                inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pd")
                text_features = model.get_text_features(**inputs)

        """
        text_outputs = self.text_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = text_outputs[1]
        text_features = paddle.matmul(pooled_output, self.text_projection)
        return text_features

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        position_ids=None,
        return_loss=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
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
                Indices of positions of each input sequence tokens in the position embeddings (TextTransformer). Selected in
                the range ``[0, max_text_length - 1]``.
                Shape as `(batch_size, num_tokens)` and dtype as int64. Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in multi-head attention (TextTransformer) to avoid performing attention on to some unwanted positions,
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
                will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`CLIPOutput` if `return_dict=True`. Otherwise it returns a tuple of tensors
            corresponding to ordered and not None (depending on the input arguments) fields of :class:`CLIPOutput`.

        Example:
            .. code-block::

                import requests
                import paddle.nn.functional as F
                from PIL import Image
                from paddlenlp.transformers import CLIPModel, CLIPProcessor

                processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
                model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

                url = "http://images.cocodataset.org/val2017/000000039769.jpg"
                image = Image.open(requests.get(url, stream=True).raw)

                inputs = processor(text=["a photo of a cat", "a photo of a dog"],
                                images=image,
                                padding=True,
                                return_tensors="pd")

                outputs = model(**inputs)

                logits_per_image = outputs[0]
                probs = F.softmax(logits_per_image, axis=1)  # we can take the softmax to get the label probabilities

        """
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
            image_embeds = paddle.matmul(vision_outputs[1], self.vision_projection)

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        text_embeds = paddle.matmul(text_outputs[1], self.text_projection)

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


class CLIPTextPretrainedModel(CLIPPretrainedModel):
    pass


@register_base_model
class CLIPTextModel(CLIPTextPretrainedModel):
    r"""
    The bare CLIPTextModel outputting :class:`BaseModelOutputWithPoolingAndCrossAttentions`.
    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        max_text_length (int, optional):
            The maximum value of the dimensionality of text position encoding, which dictates the maximum supported length of the text
            input sequence. Defaults to `64`.
        vocab_size (int, optional):
            Vocabulary size of `inputs_ids` in `CLIPModel`. Also is the vocab size of text token embedding matrix.
            Defaults to `49408`.
        text_embed_dim (int, optional):
            Dimensionality of the embedding layer and encoder layers in text model.
            Defaults to `768`.
        text_heads (int, optional):
            Number of attention heads for each attention layer in the text attention.
            Defaults to `8`.
        text_layers (int, optional):
            Number of hidden layers in the text model.
            Defaults to `12`.
        text_hidden_act (str, optional):
            The non-linear activation function of the ffn layer in the text model.
            ``"gelu"``, ``"relu"``, ``"quick_gelu"`` and any other paddle supported activation functions are supported.
            Defaults to `"quick_gelu"`.
        initializer_range (float, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Default to `0.02`.
        initializer_factor (float, optional):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing). Default to `1.`.

    """

    def __init__(
        self,
        max_text_length=77,
        text_embed_dim=512,
        text_heads=8,
        text_layers=12,
        vocab_size=49408,
        text_hidden_act="quick_gelu",
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs
    ):
        super().__init__()
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.text_embed_dim = text_embed_dim
        self.text_layers = text_layers
        self.text_model = TextTransformer(
            context_length=max_text_length,
            transformer_width=text_embed_dim,
            transformer_heads=text_heads,
            transformer_layers=text_layers,
            vocab_size=vocab_size,
            activation=text_hidden_act,
            normalize_before=True,
        )
        self.apply(self._init_weights)

    def get_input_embeddings(self) -> nn.Layer:
        return self.text_model.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.token_embedding = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        r"""
        The CLIPTextModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
                Its data type should be `int64` and it has a shape of [text_batch_size, sequence_length].
            attention_mask (Tensor, optional):
                Mask used in multi-head attention (TextTransformer) to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `0.0` values and the others have `1.0` values.
                It is a tensor with shape `[batch_size, sequence_length`.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            position_ids(Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings (TextTransformer). Selected in
                the range ``[0, max_text_length - 1]``.
                Shape as `(batch_size, num_tokens)` and dtype as int64. Defaults to `None`.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`BaseModelOutputWithPoolingAndCrossAttentions` object. If `False`, the output
                will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`BaseModelOutputWithPoolingAndCrossAttentions` if `return_dict=True`. Otherwise it returns a tuple of tensors
            corresponding to ordered and not None (depending on the input arguments) fields of :class:`BaseModelOutputWithPoolingAndCrossAttentions`.

        Example:
            .. code-block::

                from paddlenlp.transformers import CLIPTokenizer, CLIPTextModel

                model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
                tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

                inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pd")
                outputs = model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                pooled_output = outputs.pooler_output

        """
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CLIPVisionPretrainedModel(CLIPPretrainedModel):
    pass


@register_base_model
class CLIPVisionModel(CLIPVisionPretrainedModel):
    r"""
    The bare CLIPVisionModel outputting :class:`BaseModelOutputWithPoolingAndCrossAttentions`.
    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        image_resolution (int, optional):
            The size (resolution) of each image.
            Defaults to `224`.
        vision_layers (int, optional):
            Number of hidden layers in the vision model.
            Defaults to `12`.
        vision_heads (int, optional):
            Number of attention heads for each attention layer in the vision attention.
            Defaults to `12`.
        vision_embed_dim (int, optional):
            Dimensionality of the embedding layer and encoder layers in vision model.
            Defaults to `768`.
        vision_patch_size(int, optional):
            The size (resolution) of each patch.
            Defaults to `32`.
        vision_mlp_ratio(int, optional):
            The ratio between dim_feedforward and vision_hidden_dim. `radio = dim_feedforward/vision_hidden_dim`
            Defaults to `4`.
        vision_hidden_act (str, optional):
            The non-linear activation function of the ffn layer in the vision model.
            ``"gelu"``, ``"relu"``, ``"quick_gelu"`` and any other paddle supported activation functions are supported.
            Defaults to `"quick_gelu"`.
        initializer_range (float, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Default to `0.02`.
        initializer_factor (float, optional):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing). Default to `1.`.

    """

    def __init__(
        self,
        image_resolution=224,
        vision_patch_size=32,
        vision_embed_dim=768,
        vision_layers=12,
        vision_heads=12,
        vision_hidden_act="quick_gelu",
        vision_mlp_ratio=4,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs
    ):
        super().__init__()
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.vision_embed_dim = vision_embed_dim
        self.vision_layers = vision_layers

        if vision_heads is None:
            vision_heads = vision_embed_dim // 64
        self.vision_model = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_embed_dim,
            layers=vision_layers,
            heads=vision_heads,
            activation=vision_hidden_act,
            mlp_ratio=vision_mlp_ratio,
            normalize_before=True,
        )

        self.apply(self._init_weights)

    def get_input_embeddings(self) -> nn.Layer:
        return self.vision_model.conv1

    def forward(
        self,
        pixel_values=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        r"""
        The CLIPVisionModel forward method, overrides the `__call__()` special method.

        Args:
            pixel_values (Tensor):
                Pixel values. Padding will be ignored by default should you provide it.
                Its data type should be `float32` and it has a shape of [image_batch_size, num_channels, height, width].
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`BaseModelOutputWithPoolingAndCrossAttentions` object. If `False`, the output
                will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`BaseModelOutputWithPoolingAndCrossAttentions` if `return_dict=True`. Otherwise it returns a tuple of tensors
            corresponding to ordered and not None (depending on the input arguments) fields of :class:`BaseModelOutputWithPoolingAndCrossAttentions`.

        Example:
            .. code-block::

            from PIL import Image
            import requests
            from paddlenlp.transformers import CLIPProcessor, CLIPVisionModel

            model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            image = Image.open(requests.get(url, stream=True).raw)
            inputs = processor(images=image, return_tensors="pd")
            outputs = model(**inputs)

            last_hidden_state = outputs.last_hidden_state
            pooled_output = outputs.pooler_output  # pooled CLS states

        """
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CLIPForImageGeneration(CLIPPretrainedModel, DiscoDiffusionMixin, StableDiffusionMixin):
    r"""
    CLIP Model with diffusion model on top.

    Args:
        clip (:class:`CLIPModel`):
            An instance of CLIPModel.
        diffusion_type (str, optional):
            The type of diffusion. Please choose in ['disco', 'stable'].
            Defaults to `disco`.
        scheduler_type  (str, optional):
            The type of scheduler. Please choose in ['pndm', 'ddim', 'k-lms'].
            Defaults to `pndm`.
    """

    def __init__(self, clip, diffusion_type="disco", scheduler_type="pndm"):
        super().__init__()
        self.clip = clip
        self.diffusion_type = diffusion_type

        if diffusion_type == "disco":
            self.unet_model = create_unet_model(
                image_size=512,
                num_channels=256,
                num_res_blocks=2,
                channel_mult="",
                learn_sigma=True,
                class_cond=False,
                attention_resolutions="32, 16, 8",
                num_heads=4,
                num_head_channels=64,
                num_heads_upsample=-1,
                use_scale_shift_norm=True,
                dropout=0.0,
                resblock_updown=True,
                use_new_attention_order=False,
            )
            self.secondary_model = create_secondary_model()

        elif diffusion_type == "stable":
            del self.clip.vision_model
            self.vae_model = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=(
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                ),
                up_block_types=(
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                ),
                block_out_channels=(128, 256, 512, 512),
                layers_per_block=2,
                act_fn="silu",
                latent_channels=4,
                sample_size=512,
            )
            if scheduler_type == "pndm":
                self.scheduler = PNDMScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    skip_prk_steps=True,
                )
            elif scheduler_type == "ddim":
                self.scheduler = DDIMScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    clip_sample=False,
                    set_alpha_to_one=False,
                )
            elif scheduler_type == "k-lms":
                self.scheduler = LMSDiscreteScheduler(
                    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
                )
            else:
                raise ValueError('scheduler_type must be in ["pndm", "ddim", "k-lms"]')

            self.unet_model = UNet2DConditionModel(
                sample_size=64,
                in_channels=4,
                out_channels=4,
                center_input_sample=False,
                flip_sin_to_cos=True,
                freq_shift=0,
                down_block_types=(
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                ),
                block_out_channels=(320, 640, 1280, 1280),
                layers_per_block=2,
                downsample_padding=1,
                mid_block_scale_factor=1,
                act_fn="silu",
                norm_num_groups=32,
                norm_eps=1e-5,
                cross_attention_dim=768,
                attention_head_dim=8,
            )
            # input_ids_uncond
            # [49406, 49407, 49407, 49407, 49407,...,49407]
            input_ids_uncond = [49406, 49407] + [49407] * (clip.config["max_text_length"] - 2)
            self.register_buffer(
                "input_ids_uncond", paddle.to_tensor([input_ids_uncond], dtype="int64"), persistable=False
            )
        else:
            raise ValueError("diffusion_type: Please choose in ['disco', 'stable']")

        # eval mode and stop all param's gradient
        self.eval()
        for param in self.parameters():
            param.stop_gradient = True

    def generate(self, *args, **kwargs):
        if self.diffusion_type == "disco":
            return self.disco_diffusion_generate(*args, **kwargs)
        else:
            return self.stable_diffusion_generate(*args, **kwargs)

    def stable_diffusion_generate(
        self,
        input_ids,
        mode="text2image",
        init_image=None,
        mask_image=None,
        seed=None,
        strength=0.8,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        latents=None,
        fp16=True,
        **kwargs
    ):
        r"""
        The CLIPForImageGeneration stable_diffusion_generate method.

        Args:
            input_ids (Tensor):
                See :class:`CLIPModel`.
            mode (str, optional):
                The mode of StableDiffusion. Support ["text2image", "image2image", "inpaint"].
            init_image (Union[str, PIL.Image.Image], optional):
                In `"image2image"` or `"inpaint"` mode, we must input the `init_image`.
                Used in `"image2image"` and  `"inpaint"` mode.
                Default to `None`.
            mask_image (Union[str, PIL.Image], optional):
                In `"inpaint"` mode, we must input the `mask_image`.
                Used in `"inpaint"` mode.
                Default to `None`.
            seed (int, optional):
                A random number seed which is used as the basis for determining the initial.
                Default to `None`.
            strength (float, optional):
                strength is a value between 0.0 and 1.0, that controls the amount of noise that is
                added to the input image. Values that approach 1.0 allow for lots of variations but
                will also produce images that are not semantically consistent with the input.
                Used in `"image2image"` and  `"inpaint"` mode.
                Default to `0.8`.
            height (int, optional):
                The height of the image you want to generate, in pixels.
                `height` have to be divisible by 64. Used in `"text2image"` mode.
                Default to `512`.
            width (int, optional):
                The height of the image you want to generate, in pixels.
                `width` have to be divisible by 64. Used in `"text2image"` mode.
                Default to `512`.
            num_inference_steps (int, optional):
                Indicates the number of steps of inference. Generally speaking, the more steps,
                the better the result, but the more steps, the longer the generation time.
                Stable Diffusion gives good results with relatively few steps, so the default
                value is set to 50. If you want faster speed, you can reduce this value,
                if you want to generate better results, you can increase this value.
                Default to `50`.
            guidance_scale (float, optional):
                `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
                of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
                corresponds to doing no classifier free guidance.
                Default to `7.5`.
            eta (float, optional):
                eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
                eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502.
                Default to `0.0`.
            latents (paddle.Tensor, optional):
                We can specify the latents. latents_shape should be `[batch_size, unet_model.in_channels, height // 8, width // 8]`
                Default to `None`.
            fp16 (bool, optional):
                Whether to use fp16 for inference.
                Default to `True`.

        """
        assert input_ids is not None, "text2image/image2image/inpaint, please specify `input_ids`"
        if mode == "text2image":
            return self.stable_diffusion_text2image(
                input_ids=input_ids,
                seed=seed,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                latents=latents,
                fp16=fp16,
            )
        elif mode == "image2image":
            assert init_image is not None, "image2image mode, please specify `init_image`"
            # preprocess image
            init_image = self.preprocess_image(init_image)
            return self.stable_diffusion_image2image(
                input_ids=input_ids,
                init_image=init_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                seed=seed,
                fp16=fp16,
            )

        elif mode == "inpaint":
            assert init_image is not None, "inpaint mode, please specify `init_image`"
            assert mask_image is not None, "inpaint mode, please specify `mask_image`"
            # preprocess image
            init_image = self.preprocess_image(init_image)
            # preprocess mask
            mask_image = self.preprocess_mask(mask_image)

            return self.stable_diffusion_inpainting(
                input_ids=input_ids,
                init_image=init_image,
                mask_image=mask_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                seed=seed,
                fp16=fp16,
            )
        else:
            raise ValueError('Mode must be in ["text2image", "image2image", "inpaint"]')

    def disco_diffusion_generate(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        init_image=None,
        output_dir="disco_diffusion_clip_vitb32_out/",
        width_height=[1280, 768],
        skip_steps=0,
        steps=250,
        cut_ic_pow=1,
        init_scale=1000,
        clip_guidance_scale=5000,
        tv_scale=0,
        range_scale=0,
        sat_scale=0,
        cutn_batches=4,
        perlin_init=False,
        perlin_mode="mixed",
        seed=None,
        eta=0.8,
        clamp_grad=True,
        clamp_max=0.05,
        cut_overview="[12]*400+[4]*600",
        cut_innercut="[4]*400+[12]*600",
        cut_icgray_p="[0.2]*400+[0]*600",
        save_rate=10,
        n_batches=1,
        batch_name="",
        use_secondary_model=True,
        randomize_class=True,
        clip_denoised=False,
    ):
        r"""
        The CLIPForImageGeneration disco_diffusion_generate method.

        Args:
            input_ids (Tensor):
                See :class:`CLIPModel`.
            attention_mask (Tensor, optional):
                See :class:`CLIPModel`.
            position_ids (Tensor, optional):
                See :class:`CLIPModel`.
            init_image (Path, optional):
                Recall that in the image sequence above, the first image shown is just noise.  If an init_image
                is provided, diffusion will replace the noise with the init_image as its starting state.  To use
                an init_image, upload the image to the Colab instance or your Google Drive, and enter the full
                image path here. If using an init_image, you may need to increase skip_steps to ~ 50% of total
                steps to retain the character of the init. See skip_steps above for further discussion.
                Default to `None`.
            output_dir (Path, optional):
                Output directory.
                Default to `disco_diffusion_clip_vitb32_out`.
            width_height (List[int, int], optional):
                Desired final image size, in pixels. You can have a square, wide, or tall image, but each edge
                length should be set to a multiple of 64px, and a minimum of 512px on the default CLIP model setting.
                If you forget to use multiples of 64px in your dimensions, DD will adjust the dimensions of your
                image to make it so.
                Default to `[1280, 768]`.
            skip_steps (int, optional):
                Consider the chart shown here.  Noise scheduling (denoise strength) starts very high and progressively
                gets lower and lower as diffusion steps progress. The noise levels in the first few steps are very high,
                so images change dramatically in early steps.As DD moves along the curve, noise levels (and thus the
                amount an image changes per step) declines, and image coherence from one step to the next increases.
                The first few steps of denoising are often so dramatic that some steps (maybe 10-15% of total) can be
                skipped without affecting the final image. You can experiment with this as a way to cut render times.
                If you skip too many steps, however, the remaining noise may not be high enough to generate new content,
                and thus may not have time left to finish an image satisfactorily.Also, depending on your other settings,
                you may need to skip steps to prevent CLIP from overshooting your goal, resulting in blown out colors
                (hyper saturated, solid white, or solid black regions) or otherwise poor image quality.  Consider that
                the denoising process is at its strongest in the early steps, so skipping steps can sometimes mitigate
                other problems.Lastly, if using an init_image, you will need to skip ~50% of the diffusion steps to retain
                the shapes in the original init image. However, if you're using an init_image, you can also adjust
                skip_steps up or down for creative reasons.  With low skip_steps you can get a result "inspired by"
                the init_image which will retain the colors and rough layout and shapes but look quite different.
                With high skip_steps you can preserve most of the init_image contents and just do fine tuning of the texture.
                Default to `10`.
            steps:
                When creating an image, the denoising curve is subdivided into steps for processing. Each step (or iteration)
                involves the AI looking at subsets of the image called 'cuts' and calculating the 'direction' the image
                should be guided to be more like the prompt. Then it adjusts the image with the help of the diffusion denoiser,
                and moves to the next step.Increasing steps will provide more opportunities for the AI to adjust the image,
                and each adjustment will be smaller, and thus will yield a more precise, detailed image.  Increasing steps
                comes at the expense of longer render times.  Also, while increasing steps should generally increase image
                quality, there is a diminishing return on additional steps beyond 250 - 500 steps.  However, some intricate
                images can take 1000, 2000, or more steps.  It is really up to the user.  Just know that the render time is
                directly related to the number of steps, and many other parameters have a major impact on image quality, without
                costing additional time.
            cut_ic_pow (int, optional):
                This sets the size of the border used for inner cuts.  High cut_ic_pow values have larger borders, and
                therefore the cuts themselves will be smaller and provide finer details.  If you have too many or too-small
                inner cuts, you may lose overall image coherency and/or it may cause an undesirable 'mosaic' effect.
                Low cut_ic_pow values will allow the inner cuts to be larger, helping image coherency while still helping
                with some details.
                Default to `1`.
            init_scale (int, optional):
                This controls how strongly CLIP will try to match the init_image provided.  This is balanced against the
                clip_guidance_scale (CGS) above.  Too much init scale, and the image won't change much during diffusion.
                Too much CGS and the init image will be lost.
                Default to `1000`.
            clip_guidance_scale (int, optional):
                CGS is one of the most important parameters you will use. It tells DD how strongly you want CLIP to move
                toward your prompt each timestep.  Higher is generally better, but if CGS is too strong it will overshoot
                the goal and distort the image. So a happy medium is needed, and it takes experience to learn how to adjust
                CGS. Note that this parameter generally scales with image dimensions. In other words, if you increase your
                total dimensions by 50% (e.g. a change from 512 x 512 to 512 x 768), then to maintain the same effect on the
                image, you'd want to increase clip_guidance_scale from 5000 to 7500. Of the basic settings, clip_guidance_scale,
                steps and skip_steps are the most important contributors to image quality, so learn them well.
                Default to `5000`.
            tv_scale (int, optional):
                Total variance denoising. Optional, set to zero to turn off. Controls smoothness of final output. If used,
                tv_scale will try to smooth out your final image to reduce overall noise. If your image is too 'crunchy',
                increase tv_scale. TV denoising is good at preserving edges while smoothing away noise in flat regions.
                See https://en.wikipedia.org/wiki/Total_variation_denoising
                Default to `0`.
            range_scale (int, optional):
                Optional, set to zero to turn off.  Used for adjustment of color contrast.  Lower range_scale will increase
                contrast. Very low numbers create a reduced color palette, resulting in more vibrant or poster-like images.
                Higher range_scale will reduce contrast, for more muted images.
                Default to `0`.
            sat_scale (int, optional):
                Saturation scale. Optional, set to zero to turn off.  If used, sat_scale will help mitigate oversaturation.
                If your image is too saturated, increase sat_scale to reduce the saturation.
                Default to `0`.
            cutn_batches (int, optional):
                Each iteration, the AI cuts the image into smaller pieces known as cuts, and compares each cut to the prompt
                to decide how to guide the next diffusion step.  More cuts can generally lead to better images, since DD has
                more chances to fine-tune the image precision in each timestep.  Additional cuts are memory intensive, however,
                and if DD tries to evaluate too many cuts at once, it can run out of memory.  You can use cutn_batches to increase
                cuts per timestep without increasing memory usage. At the default settings, DD is scheduled to do 16 cuts per
                timestep.  If cutn_batches is set to 1, there will indeed only be 16 cuts total per timestep. However, if
                cutn_batches is increased to 4, DD will do 64 cuts total in each timestep, divided into 4 sequential batches
                of 16 cuts each.  Because the cuts are being evaluated only 16 at a time, DD uses the memory required for only 16 cuts,
                but gives you the quality benefit of 64 cuts.  The tradeoff, of course, is that this will take ~4 times as long to
                render each image.So, (scheduled cuts) x (cutn_batches) = (total cuts per timestep). Increasing cutn_batches will
                increase render times, however, as the work is being done sequentially.  DD's default cut schedule is a good place
                to start, but the cut schedule can be adjusted in the Cutn Scheduling section, explained below.
                Default to `4`.
            perlin_init (bool, optional):
                Normally, DD will use an image filled with random noise as a starting point for the diffusion curve.
                If perlin_init is selected, DD will instead use a Perlin noise model as an initial state.  Perlin has very
                interesting characteristics, distinct from random noise, so it's worth experimenting with this for your projects.
                Beyond perlin, you can, of course, generate your own noise images (such as with GIMP, etc) and use them as an
                init_image (without skipping steps). Choosing perlin_init does not affect the actual diffusion process, just the
                starting point for the diffusion. Please note that selecting a perlin_init will replace and override any init_image
                you may have specified. Further, because the 2D, 3D and video animation systems all rely on the init_image system,
                if you enable Perlin while using animation modes, the perlin_init will jump in front of any previous image or video
                input, and DD will NOT give you the expected sequence of coherent images. All of that said, using Perlin and
                animation modes together do make a very colorful rainbow effect, which can be used creatively.
                Default to `False`.
            perlin_mode (str, optional):
                sets type of Perlin noise: colored, gray, or a mix of both, giving you additional options for noise types. Experiment
                to see what these do in your projects.
                Default to `mixed`.
            seed (int, optional):
                Deep in the diffusion code, there is a random number seed which is used as the basis for determining the initial
                state of the diffusion.  By default, this is random, but you can also specify your own seed. This is useful if you like a
                particular result and would like to run more iterations that will be similar. After each run, the actual seed value used will be
                reported in the parameters report, and can be reused if desired by entering seed # here.  If a specific numerical seed is used
                repeatedly, the resulting images will be quite similar but not identical.
                Default to `None`.
            eta (float, optional):
                Eta (greek letter η) is a diffusion model variable that mixes in a random amount of scaled noise into each timestep.
                0 is no noise, 1.0 is more noise. As with most DD parameters, you can go below zero for eta, but it may give you
                unpredictable results. The steps parameter has a close relationship with the eta parameter. If you set eta to 0,
                then you can get decent output with only 50-75 steps. Setting eta to 1.0 favors higher step counts, ideally around
                250 and up. eta has a subtle, unpredictable effect on image, so you'll need to experiment to see how this affects your projects.
                Default to `0.8`.
            clamp_grad (bool, optional):
                As I understand it, clamp_grad is an internal limiter that stops DD from producing extreme results. Try your images with and without
                clamp_grad. If the image changes drastically with clamp_grad turned off, it probably means your clip_guidance_scale is too high and
                should be reduced.
                Default to `True`.
            clamp_max (float, optional):
                Sets the value of the clamp_grad limitation. Default is 0.05, providing for smoother, more muted coloration in images, but setting
                higher values (0.15-0.3) can provide interesting contrast and vibrancy.
                Default to `0.05`.
            cut_overview (str, optional):
                The schedule of overview cuts.
                Default to `'[12]*400+[4]*600'`.
            cut_innercut (str, optional):
                The schedule of inner cuts.
                Default to `'[4]*400+[12]*600'`.
            cut_icgray_p (str, optional):
                This sets the size of the border used for inner cuts.  High cut_ic_pow values have larger borders, and therefore the cuts
                themselves will be smaller and provide finer details.  If you have too many or too-small inner cuts, you may lose overall
                image coherency and/or it may cause an undesirable 'mosaic' effect.   Low cut_ic_pow values will allow the inner cuts to be
                larger, helping image coherency while still helping with some details.
                Default to `'[0.2]*400+[0]*600'`.
            save_rate (int, optional):
                During a diffusion run, you can monitor the progress of each image being created with this variable.  If display_rate is set
                to 50, DD will show you the in-progress image every 50 timesteps. Setting this to a lower value, like 5 or 10, is a good way
                to get an early peek at where your image is heading. If you don't like the progression, just interrupt execution, change some
                settings, and re-run.  If you are planning a long, unmonitored batch, it's better to set display_rate equal to steps, because
                displaying interim images does slow Colab down slightly.
                Default to `10`.
            n_batches (int, optional):
                This variable sets the number of still images you want DD to create.  If you are using an animation mode (see below for details)
                DD will ignore n_batches and create a single set of animated frames based on the animation settings.
                Default to `1`.
            batch_name (str, optional):
                The name of the batch, the batch id will be named as "progress-[batch_name]-seed-[range(n_batches)]-[save_rate]". To avoid your
                artworks be overridden by other users, please use a unique name.
                Default to `''`.
            use_secondary_model (bool, optional):
                Whether or not use secondary model.
                Default to `True`.
            randomize_class (bool, optional):
                Random class.
                Default to `True`.
            clip_denoised (bool, optional):
                Clip denoised.
                Default to `False`.

        Returns:
            List[PIL.Image]: Returns n_batches of final image.
            Its data type should be PIL.Image.

        Example:
            .. code-block::

            from paddlenlp.transformers import CLIPForImageGeneration, CLIPTokenizer

            # Initialize the model and tokenizer
            model_name_or_path = 'openai/disco-diffusion-clip-vit-base-patch32'
            model = CLIPForImageGeneration.from_pretrained(model_name_or_path)
            tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path)
            model.eval()

            # Prepare the text_prompt.
            text_prompt = "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation."
            # Style and artist can be specified
            style = None
            artist = None
            text_prompt = model.preprocess_text_prompt(text_prompt, style=style, artist=artist)

            # CLIP's pad_token_id is 0 and padding to max length 77 (tokenizer.model_max_length).
            raw_pad_token_id = tokenizer.pad_token_id
            tokenizer.pad_token_id = 0
            tokenized_inputs = tokenizer(text_prompt, return_tensors="pd", padding="max_length", max_length=tokenizer.model_max_length)
            images = model.generate(**tokenized_inputs)

            # return List[PIL.Image]
            images[0].save("figure.png")

        """
        self.diffusion = create_gaussian_diffusion(
            steps=steps,
            learn_sigma=True,
            sigma_small=False,
            noise_schedule="linear",
            predict_xstart=False,
            rescale_timesteps=True,
        )
        # get get_text_features
        target_text_embeds = self.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
        )

        images_list = super().disco_diffusion_generate(
            target_text_embeds=target_text_embeds,
            clamp_grad=clamp_grad,
            clamp_max=clamp_max,
            clip_denoised=clip_denoised,
            clip_guidance_scale=clip_guidance_scale,
            cut_ic_pow=cut_ic_pow,
            cut_icgray_p=cut_icgray_p,
            cut_innercut=cut_innercut,
            cut_overview=cut_overview,
            cutn_batches=cutn_batches,
            save_rate=save_rate,
            eta=eta,
            init_image=init_image,
            init_scale=init_scale,
            n_batches=n_batches,
            output_dir=output_dir,
            perlin_init=perlin_init,
            perlin_mode=perlin_mode,
            randomize_class=randomize_class,
            range_scale=range_scale,
            sat_scale=sat_scale,
            seed=seed,
            skip_steps=skip_steps,
            tv_scale=tv_scale,
            use_secondary_model=use_secondary_model,
            width_height=width_height,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            batch_name=batch_name,
        )

        return images_list

    def preprocess_text_prompt(self, text_prompt, style=None, artist=None):
        text_prompt = text_prompt.rstrip(",.，。")
        if style is not None:
            text_prompt += ",{}".format(style)
        if artist is not None:
            text_prompt += ",{},trending on artstation".format(artist)
        return text_prompt

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
