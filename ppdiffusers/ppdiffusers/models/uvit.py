# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import einops
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Constant, TruncatedNormal

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from .modeling_utils import ModelMixin
from .unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block

# Common initializations
ones_ = Constant(value=1.0)
zeros_ = Constant(value=0.0)
trunc_normal_ = TruncatedNormal(std=0.02)


def drop_path(input, drop_prob: float = 0.0, training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + paddle.rand(shape, dtype=input.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = (input / keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
    return:
        embedding: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = paddle.exp(-math.log(max_period) * paddle.arange(start=0, end=half, dtype=paddle.float32) / half)
    args = timesteps[:, None].astype("float32") * freqs[None]
    embedding = paddle.concat([paddle.cos(args), paddle.sin(args)], axis=-1)
    if dim % 2:
        embedding = paddle.concat([embedding, paddle.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


def unpatchify(x, in_chans):
    patch_size = int((x.shape[2] // in_chans) ** 0.5)
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1] and patch_size ** 2 * in_chans == x.shape[2]
    x = einops.rearrange(x, "B (h w) (p1 p2 C) -> B C (h p1) (w p2)", h=h, p1=patch_size, p2=patch_size)
    return x


def interpolate_pos_emb(pos_emb, old_shape, new_shape):
    pos_emb = einops.rearrange(pos_emb, "B (H W) C -> B C H W", H=old_shape[0], W=old_shape[1])
    pos_emb = F.interpolate(pos_emb, new_shape, mode="bilinear")
    pos_emb = einops.rearrange(pos_emb, "B C H W -> B (H W) C")
    return pos_emb


class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x)
        # TODO: xformers support
        with paddle.amp.auto_cast(enable=False):
            qkv = einops.rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads).astype("float32")
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = paddle.mm(q, k.transpose([0, 1, 3, 2])) * self.scale  # B H L L
            attn = F.softmax(attn, axis=-1)
            attn = self.attn_drop(attn)
            x = paddle.mm(attn, v).transpose([0, 2, 1, 3]).reshape([B, L, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        skip=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim) if skip else None
        self.norm2 = norm_layer(dim)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:  # TODO: use_checkpoint
            return paddle.utils.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(paddle.concat([x, skip], axis=-1))
            x = self.norm1(x)
        x = x + self.drop_path(self.attn(x))
        x = self.norm2(x)

        x = x + self.drop_path(self.mlp(x))
        x = self.norm3(x)

        return x


class PatchEmbed(nn.Layer):
    """Image to Patch Embedding"""

    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        _, _, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose([0, 2, 1])
        return x


@dataclass
class UViTModelOutput(BaseOutput):
    """
    Args:
        sample (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states output. Output of last layer of model.
    """

    sample_img: paddle.Tensor
    sample_clip_img: paddle.Tensor
    sample_text: paddle.Tensor


# class UViTModel(ModelMixin, ConfigMixin):
class UViTModel(nn.Layer):
    r"""
    UViTModel is a unet-stype ViT model that takes in a noisy sample and a timestep and returns sample shaped output.
    Note that the different is the

    """

    # @register_to_config
    def __init__(
        self,
        img_size=64,
        in_chans=4,
        patch_size=2,
        embed_dim=1536,
        depth=30,
        num_heads=24,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        pos_drop_rate=0.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        mlp_time_embed=False,
        use_checkpoint=False,
        text_dim=64,
        num_text_tokens=77,
        clip_img_dim=512,
        pretrained_path=None,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size  # the default img size
        assert self.img_size[0] % patch_size == 0 and self.img_size[1] % patch_size == 0
        self.num_patches = (self.img_size[0] // patch_size) * (self.img_size[1] // patch_size)

        self.time_img_embed = (
            nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.SiLU(),
                nn.Linear(4 * embed_dim, embed_dim),
            )
            if mlp_time_embed
            else nn.Identity()
        )

        self.time_text_embed = (
            nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.SiLU(),
                nn.Linear(4 * embed_dim, embed_dim),
            )
            if mlp_time_embed
            else nn.Identity()
        )

        self.text_embed = nn.Linear(text_dim, embed_dim)
        self.text_out = nn.Linear(embed_dim, text_dim)

        self.clip_img_embed = nn.Linear(clip_img_dim, embed_dim)
        self.clip_img_out = nn.Linear(embed_dim, clip_img_dim)

        self.num_text_tokens = num_text_tokens
        self.num_tokens = 1 + 1 + num_text_tokens + 1 + self.num_patches

        self.pos_embed = self.create_parameter(
            shape=(1, self.num_tokens, embed_dim), default_initializer=Constant(value=0.0)
        )

        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        self.in_blocks = nn.LayerList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for _ in range(depth // 2)
            ]
        )

        self.mid_block = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )

        self.out_blocks = nn.LayerList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    skip=True,
                    use_checkpoint=use_checkpoint,
                )
                for _ in range(depth // 2)
            ]
        )

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size**2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias_attr=True)

        trunc_normal_(self.pos_embed)
        self.apply(self._init_weights)

        self.token_embedding = nn.Embedding(2, embed_dim)
        self.pos_embed_token = self.create_parameter(shape=(1, 1, embed_dim), default_initializer=Constant(value=0.0))

        if pretrained_path:
            self.set_dict(paddle.load(pretrained_path))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def no_weight_decay(self):
        return {"pos_embed"}

    def forward(
        self,
        img,
        clip_img,
        text,
        t_img,
        t_text,
        data_type,
        return_dict=True,
    ):
        _, _, H, W = img.shape
        img = self.patch_embed(img)
        clip_img = self.clip_img_embed(clip_img)
        text = self.text_embed(text)

        t_img_token = self.time_img_embed(timestep_embedding(t_img, self.embed_dim)).unsqueeze(axis=1)
        t_text_token = self.time_text_embed(timestep_embedding(t_text, self.embed_dim)).unsqueeze(axis=1)
        token_embed = self.token_embedding(data_type).unsqueeze(axis=1)

        x = paddle.concat((t_img_token, t_text_token, token_embed, text, clip_img, img), axis=1)

        num_text_tokens, num_img_tokens = text.shape[1], img.shape[1]

        pos_embed = paddle.concat(
            [self.pos_embed[:, : 1 + 1, :], self.pos_embed_token, self.pos_embed[:, 1 + 1 :, :]], axis=1
        )

        if H == self.img_size[0] and W == self.img_size[1]:
            pass
        else:  # interpolate the positional embedding when the input image is not of the default shape
            pos_embed_others, pos_embed_patches = paddle.split(
                pos_embed, [1 + 1 + 1 + num_text_tokens + 1, self.num_patches], axis=1
            )
            pos_embed_patches = interpolate_pos_emb(
                pos_embed_patches,
                (self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size),
                (H // self.patch_size, W // self.patch_size),
            )
            pos_embed = paddle.concat((pos_embed_others, pos_embed_patches), axis=1)

        x = x + pos_embed
        x = self.pos_drop(x)

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)

        t_img_token_out, t_text_token_out, token_embed_out, text_out, clip_img_out, img_out = x.split(
            (1, 1, 1, num_text_tokens, 1, num_img_tokens), axis=1
        )

        img_out = self.decoder_pred(img_out)
        sample_img = unpatchify(img_out, self.in_chans)
        sample_clip_img = self.clip_img_out(clip_img_out)
        sample_text = self.text_out(text_out)

        return sample_img, sample_clip_img, sample_text

        # if not return_dict:
        #     return (sample_img, sample_clip_img, sample_text)

        # return UViTModelOutput(sample_img=sample_img, sample_clip_img=sample_clip_img, sample_text=sample_text)
