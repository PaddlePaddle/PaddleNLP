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
from dataclasses import dataclass

import einops
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .attention import DropPath, Mlp
from .embeddings import PatchEmbed, get_timestep_embedding
from .modeling_utils import ModelMixin


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

    def forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(paddle.concat([x, skip], axis=-1))
            x = self.norm1(x)
        x = x + self.drop_path(self.attn(x))
        x = self.norm2(x)

        x = x + self.drop_path(self.mlp(x))
        x = self.norm3(x)

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


class UViTModel(ModelMixin, ConfigMixin):
    r"""
    UViTModel is a unet-stype ViT model that takes in a noisy sample and a timestep and returns sample shaped output.
    Note that the different between the original U-ViT is the post-layer normalization and add a layer normalization
    after concatenat-ing a long skip connection, which stabilizes the training of U-ViT in UniDiffuser.

    """

    @register_to_config
    def __init__(
        self,
        sample_size=1,
        img_size=64,
        in_channels=4,
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
        text_dim=64,
        num_text_tokens=77,
        clip_img_dim=512,
        use_checkpoint=False,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_embed = PatchEmbed(
            height=self.img_size[0],
            width=self.img_size[1],
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            add_pos_embed=False,
        )
        assert self.img_size[0] % patch_size == 0 and self.img_size[1] % patch_size == 0
        self.num_patches = (self.img_size[0] // patch_size) * (self.img_size[1] // patch_size)

        self.encode_prefix = nn.Linear(768, text_dim)

        self.text_embed = nn.Linear(text_dim, embed_dim)
        self.text_out = nn.Linear(embed_dim, text_dim)
        self.clip_img_embed = nn.Linear(clip_img_dim, embed_dim)
        self.clip_img_out = nn.Linear(embed_dim, clip_img_dim)

        self.num_text_tokens = num_text_tokens
        self.num_tokens = 1 + 1 + num_text_tokens + 1 + self.num_patches

        self.pos_embed = self.create_parameter(
            shape=(1, self.num_tokens, embed_dim),
            default_initializer=nn.initializer.Constant(0.0),
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
                )
                for _ in range(depth // 2)
            ]
        )

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size**2 * in_channels
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias_attr=True)

        self.token_embedding = nn.Embedding(2, embed_dim)
        self.pos_embed_token = self.create_parameter(
            shape=(1, 1, embed_dim), default_initializer=nn.initializer.Constant(0.0)
        )

    def forward(
        self,
        img: paddle.Tensor,
        clip_img: paddle.Tensor,
        text: paddle.Tensor,
        t_img: paddle.Tensor,
        t_text: paddle.Tensor,
        data_type: paddle.Tensor,
        return_dict=False,  # TODO: nf
    ):
        _, _, H, W = img.shape
        img = self.patch_embed(img)
        clip_img = self.clip_img_embed(clip_img)
        text = self.text_embed(text)

        t_img_token = get_timestep_embedding(t_img, self.embed_dim, True, 0).unsqueeze(axis=1)
        t_text_token = get_timestep_embedding(t_text, self.embed_dim, True, 0).unsqueeze(axis=1)
        token_embed = self.token_embedding(data_type).unsqueeze(axis=1)

        x = paddle.concat((t_img_token, t_text_token, token_embed, text, clip_img, img), axis=1)

        num_text_tokens, num_img_tokens = text.shape[1], img.shape[1]

        pos_embed = paddle.concat(
            [self.pos_embed[:, : 1 + 1, :], self.pos_embed_token, self.pos_embed[:, 1 + 1 :, :]], axis=1
        )

        if H == self.img_size[0] and W == self.img_size[1]:
            pass
        else:
            # interpolate the positional embedding when the input image is not of the default shape
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
        sample_img = unpatchify(img_out, self.in_channels)
        sample_clip_img = self.clip_img_out(clip_img_out)
        sample_text = self.text_out(text_out)

        if not return_dict:
            return (sample_img, sample_clip_img, sample_text)

        return UViTModelOutput(sample_img=sample_img, sample_clip_img=sample_clip_img, sample_text=sample_text)
