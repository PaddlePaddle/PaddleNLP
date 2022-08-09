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

from typing import Union, Tuple, Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .. import PretrainedModel, register_base_model


# set attr
def quick_gelu(x):
    return x * F.sigmoid(1.702 * x)


F.quick_gelu = quick_gelu

INF = float("-inf")  # -1e4 -1e9


class ModifiedResNet(nn.Layer):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self,
                 layers,
                 output_dim,
                 heads,
                 input_resolution=224,
                 width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2D(3,
                               width // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(width // 2)
        self.conv2 = nn.Conv2D(width // 2,
                               width // 2,
                               kernel_size=3,
                               padding=1,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(width // 2)
        self.conv3 = nn.Conv2D(width // 2,
                               width,
                               kernel_size=3,
                               padding=1,
                               bias_attr=False)
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
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim,
                                        heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2),
                             (self.conv3, self.bn3)]:
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


def multi_head_attention_forward(x: paddle.Tensor,
                                 num_heads: int,
                                 q_proj: nn.Linear,
                                 k_proj: nn.Linear,
                                 v_proj: nn.Linear,
                                 c_proj: nn.Linear,
                                 attn_mask: Optional[paddle.Tensor] = None):
    max_len, batch_size, emb_dim = x.shape
    head_dim = emb_dim // num_heads
    scaling = float(head_dim)**-0.5
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
        assert attn_mask.shape[0] == 1 and attn_mask.shape[
            1] == max_len and attn_mask.shape[2] == max_len
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

        self.conv3 = nn.Conv2D(planes,
                               planes * self.expansion,
                               1,
                               bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(
                ("-1", nn.AvgPool2D(stride)),
                ("0",
                 nn.Conv2D(inplanes,
                           planes * self.expansion,
                           1,
                           stride=1,
                           bias_attr=False)),
                ("1", nn.BatchNorm2D(planes * self.expansion)))

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

    def __init__(self,
                 spacial_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 output_dim: int = None):
        super().__init__()

        self.positional_embedding = paddle.create_parameter(
            (spacial_dim**2 + 1, embed_dim), dtype=paddle.get_default_dtype())

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias_attr=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias_attr=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias_attr=True)
        self.c_proj = nn.Linear(embed_dim,
                                output_dim or embed_dim,
                                bias_attr=True)
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

    def forward(self, x):

        x = x.reshape(
            (x.shape[0], x.shape[1], x.shape[2] * x.shape[3])).transpose(
                (2, 0, 1))  # NCHW -> (HW)NC
        x = paddle.concat([x.mean(axis=0, keepdim=True), x], axis=0)
        x = x + paddle.unsqueeze(self.positional_embedding, 1)
        out = multi_head_attention_forward(x, self.num_heads, self.q_proj,
                                           self.k_proj, self.v_proj,
                                           self.c_proj)

        return out[0]


class VisualTransformer(nn.Layer):

    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 activation: str,
                 normalize_before: bool = True):
        super().__init__()
        # used patch_size x patch_size, stride patch_size to do linear projection
        self.conv1 = nn.Conv2D(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias_attr=False)

        self.class_embedding = paddle.create_parameter(
            (width, ), paddle.get_default_dtype())

        self.positional_embedding = paddle.create_parameter(
            ((input_resolution // patch_size)**2 + 1, width),
            paddle.get_default_dtype())

        self.ln_pre = nn.LayerNorm(width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            normalize_before=normalize_before,
            dropout=0,
            activation=activation,
            attn_dropout=0,
            act_dropout=0)
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)

        self.ln_post = nn.LayerNorm(width)

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = x.transpose((0, 2, 1))
        x = paddle.concat([
            self.class_embedding.unsqueeze([0, 1]).expand([x.shape[0], -1, -1]),
            x
        ],
                          axis=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x[:, 0])

        return x


class TextTransformer(nn.Layer):

    def __init__(self,
                 context_length,
                 transformer_width,
                 transformer_heads,
                 transformer_layers,
                 vocab_size,
                 activation="quick_gelu",
                 normalize_before=True):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_width,
            nhead=transformer_heads,
            dim_feedforward=transformer_width * 4,
            normalize_before=normalize_before,
            dropout=0,
            activation=activation,
            attn_dropout=0,
            act_dropout=0)
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 transformer_layers)

        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = paddle.create_parameter(
            (context_length, transformer_width), paddle.get_default_dtype())
        self.ln_final = nn.LayerNorm(transformer_width)

        self.register_buffer("attention_mask",
                             paddle.triu(paddle.ones(
                                 (1, 1, context_length, context_length)) * INF,
                                         diagonal=1),
                             persistable=False)

    def forward(self, text):
        seqlen = text.shape[1]
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding[:seqlen]
        x = self.transformer(
            x, src_mask=self.attention_mask[:, :, :seqlen, :seqlen])

        x = self.ln_final(x)

        x = x.gather_nd(
            paddle.stack([paddle.arange(x.shape[0]),
                          text.argmax(-1)], axis=-1))
        return x


class ClipPreTrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "openai/clip-vit-base-patch32": {
            "embed_dim": 512,
            # vision
            "image_resolution": 224,
            "vision_layers": 12,
            "vision_width": 768,
            "vision_patch_size": 32,
            # text
            "context_length": 77,
            "vocab_size": 49408,
            "transformer_width": 512,
            "transformer_heads": 8,
            "transformer_layers": 12,
            "initializer_range": 0.02,
            "hidden_act": "quick_gelu",
            "logit_scale_init_value": 2.6592
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "openai/clip-vit-base-patch32": {}
        }
    }
    base_model_prefix = "clip"


@register_base_model
class CLIPModel(ClipPreTrainedModel):

    def __init__(
            self,
            embed_dim: int = 512,
            # vision
            image_resolution: int = 224,
            vision_layers: Union[Tuple[int, int, int, int], int] = 12,
            vision_width: int = 768,
            vision_patch_size: int = 32,
            # text
            context_length: int = 77,
            vocab_size: int = 49408,
            transformer_width: int = 512,
            transformer_heads: int = 8,
            transformer_layers: int = 12,
            initializer_range: float = 0.02,
            hidden_act: str = "quick_gelu",
            logit_scale_init_value: float = 2.6592):
        super().__init__()
        self.initializer_range = initializer_range
        self.logit_scale_init_value = logit_scale_init_value
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(layers=vision_layers,
                                         output_dim=embed_dim,
                                         heads=vision_heads,
                                         input_resolution=image_resolution,
                                         width=vision_width)
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(input_resolution=image_resolution,
                                            patch_size=vision_patch_size,
                                            width=vision_width,
                                            layers=vision_layers,
                                            heads=vision_heads,
                                            activation=hidden_act,
                                            normalize_before=True)

        self.text = TextTransformer(context_length=context_length,
                                    transformer_width=transformer_width,
                                    transformer_heads=transformer_heads,
                                    transformer_layers=transformer_layers,
                                    vocab_size=vocab_size,
                                    activation=hidden_act,
                                    normalize_before=True)

        self.visual_projection = paddle.create_parameter(
            (vision_width, embed_dim), paddle.get_default_dtype())
        self.text_projection = paddle.create_parameter(
            (transformer_width, embed_dim), paddle.get_default_dtype())

        self.logit_scale = paddle.create_parameter(
            (1, ),
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(logit_scale_init_value))

    def encode_image(self, pixel_values):
        return self.visual_projection(self.visual(pixel_values))

    def encode_text(self, input_ids):
        return self.text_projection(self.text(input_ids))

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        image_features = self.encode_image(pixel_values)
        text_features = self.encode_text(input_ids)

        # normalized features
        image_features = image_features / image_features.norm(axis=-1,
                                                              keepdim=True)
        text_features = text_features / text_features.norm(axis=-1,
                                                           keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = paddle.matmul(logit_scale * image_features,
                                         text_features,
                                         transpose_y=True)
        logits_per_text = paddle.matmul(logit_scale * text_features,
                                        image_features,
                                        transpose_y=True)

        return logits_per_image, logits_per_text
