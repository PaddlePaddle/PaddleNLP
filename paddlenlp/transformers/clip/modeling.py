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

__all__ = [
    'VisualTransformer',
    'TextTransformer',
    'CLIPPreTrainedModel',
]


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


class CLIPPreTrainedModel(PretrainedModel):
    """
    An abstract class for pretrained CLIP models. It provides CLIP related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "openai/clip-vit-base-patch32": {
            # vision
            "image_resolution": 224,
            "vision_layers": 12,
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
            "logit_scale_init_value": 2.6592
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "openai/clip-vit-base-patch32":
            "http://bj.bcebos.com/paddlenlp/models/transformers/openai/clip-vit-base-patch32/model_state.pdparams"
        }
    }
    base_model_prefix = "clip"

    def _init_weights(self, layer):
        """Initialize the weights"""
        initializer_range = self.initializer_range if hasattr(
            self,
            "initializer_range") else self.clip.config["initializer_range"]
        factor = self.initializer_factor if hasattr(
            self,
            "initializer_factor") else self.clip.config["initializer_factor"]
        vision_embed_dim = self.vision_embed_dim if hasattr(
            self, "vision_embed_dim") else self.clip.config["vision_embed_dim"]
        text_embed_dim = self.text_embed_dim if hasattr(
            self, "text_embed_dim") else self.clip.config["text_embed_dim"]
        vision_layers = self.vision_layers if hasattr(
            self, "vision_layers") else self.clip.config["vision_layers"]
        text_layers = self.text_layers if hasattr(
            self, "text_layers") else self.clip.config["text_layers"]

        if isinstance(layer, VisualTransformer):
            # vision embedding
            layer.class_embedding.set_value(
                paddle.normal(
                    std=vision_embed_dim**-0.5 * factor,
                    shape=layer.class_embedding.shape,
                ))
            layer.conv1.weight.set_value(
                paddle.normal(
                    std=initializer_range * factor,
                    shape=layer.conv1.weight.shape,
                ))
            layer.positional_embedding.set_value(
                paddle.normal(
                    std=initializer_range * factor,
                    shape=layer.positional_embedding.shape,
                ))

        elif isinstance(layer, TextTransformer):
            # text embedding
            layer.token_embedding.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * 0.02,
                    shape=layer.token_embedding.weight.shape,
                ))
            layer.positional_embedding.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * 0.02,
                    shape=layer.positional_embedding.shape,
                ))
        elif isinstance(layer, CLIPModel):
            layer.text_projection.set_value(
                paddle.normal(
                    std=text_embed_dim**-0.5 * factor,
                    shape=layer.text_projection.shape,
                ))
            layer.visual_projection.set_value(
                paddle.normal(
                    std=vision_embed_dim**-0.5 * factor,
                    shape=layer.visual_projection.shape,
                ))
            for name, sub_layer in layer.named_sublayers():
                num_layers = vision_layers if "visual_model" in name else text_layers
                if isinstance(sub_layer, nn.TransformerEncoderLayer):
                    # self_attn
                    in_proj_std = (sub_layer.self_attn.embed_dim**-0.5) * (
                        (2 * num_layers)**-0.5) * factor
                    out_proj_std = (sub_layer.self_attn.embed_dim**
                                    -0.5) * factor
                    sub_layer.self_attn.q_proj.weight.set_value(
                        paddle.normal(
                            std=in_proj_std,
                            shape=sub_layer.self_attn.q_proj.weight.shape,
                        ))
                    sub_layer.self_attn.k_proj.weight.set_value(
                        paddle.normal(
                            std=in_proj_std,
                            shape=sub_layer.self_attn.k_proj.weight.shape,
                        ))
                    sub_layer.self_attn.v_proj.weight.set_value(
                        paddle.normal(
                            std=in_proj_std,
                            shape=sub_layer.self_attn.v_proj.weight.shape,
                        ))
                    sub_layer.self_attn.out_proj.weight.set_value(
                        paddle.normal(
                            std=out_proj_std,
                            shape=sub_layer.self_attn.out_proj.weight.shape,
                        ))
                    # ffn
                    in_proj_std = ((sub_layer._config["d_model"]**-0.5) *
                                   ((2 * num_layers)**-0.5) * factor)
                    fc_std = (2 * sub_layer._config["d_model"])**-0.5 * factor
                    sub_layer.linear1.weight.set_value(
                        paddle.normal(
                            std=fc_std,
                            shape=sub_layer.linear1.weight.shape,
                        ))
                    sub_layer.linear2.weight.set_value(
                        paddle.normal(
                            std=in_proj_std,
                            shape=sub_layer.linear2.weight.shape,
                        ))
        if isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.ones_like(layer.weight))
        if isinstance(layer, nn.Linear) and layer.bias is not None:
            layer.bias.set_value(paddle.zeros_like(layer.bias))


@register_base_model
class CLIPModel(CLIPPreTrainedModel):
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
        vision_layers (Union[Tuple[int, int, int, int], int], optional):
            Number of hidden layers in the vision model.
            Defaults to `12`.
        vision_embed_dim (int, optional):
            Dimensionality of the embedding layer and encoder layers in vision model.
            Defaults to `768`.
        vision_patch_size(int, optional):
            The size (resolution) of each patch.
            Defaults to `32`.
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
            Number of attention heads for each attention layer in the attention.
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
            vision_embed_dim: int = 768,
            vision_patch_size: int = 32,
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
            logit_scale_init_value: float = 2.6592):
        super().__init__()
        self.initializer_factor = initializer_factor
        self.initializer_range = initializer_range
        self.logit_scale_init_value = logit_scale_init_value
        self.vision_embed_dim = vision_embed_dim
        self.text_embed_dim = text_embed_dim
        self.vision_layers = vision_layers
        self.text_layers = text_layers
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_embed_dim * 32 // 64
            self.visual_model = ModifiedResNet(
                layers=vision_layers,
                output_dim=projection_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_embed_dim)
            self.visual_projection = None
        else:
            vision_heads = vision_embed_dim // 64
            self.visual_model = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_embed_dim,
                layers=vision_layers,
                heads=vision_heads,
                activation=vision_hidden_act,
                normalize_before=True)
            self.visual_projection = paddle.create_parameter(
                (vision_embed_dim, projection_dim), paddle.get_default_dtype())

        self.text_model = TextTransformer(context_length=max_text_length,
                                          transformer_width=text_embed_dim,
                                          transformer_heads=text_heads,
                                          transformer_layers=text_layers,
                                          vocab_size=vocab_size,
                                          activation=text_hidden_act,
                                          normalize_before=True)

        self.text_projection = paddle.create_parameter(
            (text_embed_dim, projection_dim), paddle.get_default_dtype())

        self.logit_scale = paddle.create_parameter(
            (1, ),
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(logit_scale_init_value))
        self.apply(self._init_weights)

    def encode_image(self, pixel_values):
        if self.visual_projection is not None:
            return paddle.matmul(self.visual_model(pixel_values),
                                 self.visual_projection)
        else:
            return self.visual_model(pixel_values)

    def encode_text(self, input_ids):
        return paddle.matmul(self.text_model(input_ids), self.text_projection)

    def forward(self, input_ids, pixel_values, **kwargs):
        r'''
        The CLIPModel forward method, overrides the `__call__()` special method.
        
        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
                Its data type should be `int64` and it has a shape of [text_batch_size, sequence_length].
            pixel_values (Tensor):
                Pixel values. Padding will be ignored by default should you provide it.
                Its data type should be `float32` and it has a shape of [image_batch_size, num_channels, height, width].
                
        Returns:
            tuple: Returns tuple (`logits_per_image`, `logits_per_text`).

            With the fields:

            - `logits_per_image` (Tensor):
                The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
                similarity scores.
                Its data type should be float32 and its shape is [image_batch_size, text_batch_size].

            - `logits_per_text` (Tensor):
                The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
                similarity scores.
                Its data type should be float32 and its shape is [text_batch_size, image_batch_size].
            
        Example:
            .. code-block::
            
                import paddle
                from paddlenlp.transformers import AutoModel, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-base-patch32')
                model = AutoModel.from_pretrained('openai/clip-vit-base-patch32')
                model.eval()
                inputs = tokenizer(["a photo of a cat", "a photo of a dog"],
                                padding=True,
                                return_tensors="pd",
                                return_token_type_ids=False)
                inputs["pixel_values"] = paddle.randn((4, 3, 224, 224))
                logits_per_image, logits_per_text = model(**inputs)
                # logits_per_image's shape [4, 2]
                # logits_per_text's shape [2, 4]
        '''
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
