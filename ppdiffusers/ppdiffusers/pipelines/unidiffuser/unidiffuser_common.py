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

import random

import einops
import ml_collections
import numpy as np
import paddle
import paddle.nn as nn
from PIL import Image


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = "noise_pred"
    config.z_shape = (4, 64, 64)
    config.clip_img_dim = 512
    config.clip_text_dim = 768
    config.text_dim = 64  # reduce dimension
    config.data_type = 1

    config.autoencoder = d(
        pretrained_path="models/autoencoder_kl.pdparams",
    )

    config.caption_decoder = d(
        pretrained_path="models/caption_decoder.pdparams", hidden_dim=config.get_ref("text_dim")
    )

    config.nnet = d(
        img_size=64,
        in_chans=4,
        patch_size=2,
        embed_dim=1536,
        depth=30,
        num_heads=24,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,
        pos_drop_rate=0.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        mlp_time_embed=False,
        text_dim=config.get_ref("text_dim"),
        num_text_tokens=77,
        clip_img_dim=config.get_ref("clip_img_dim"),
        use_checkpoint=False,
    )

    config.sample = d(sample_steps=50, scale=7.0, t2i_cfg_mode="true_uncond")
    return config


def save_image(tensor, fp, format=None):
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = (tensor * 255 + 0.5).clip_(0, 255).transpose([1, 2, 0]).astype("uint8").numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


def to_pil_image(tensor, mode=None):
    ndarr = (tensor * 255 + 0.5).clip_(0, 255).transpose([1, 2, 0]).astype("uint8").numpy()
    return Image.fromarray(ndarr, mode=mode)


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = paddle.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=paddle.float64) ** 2
    return _betas.numpy()


def center_crop(width, height, img):
    resample = {"box": Image.BOX, "lanczos": Image.LANCZOS}["lanczos"]
    crop = np.min(img.shape[:2])
    img = img[
        (img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2,
        (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2,
    ]  # center crop
    try:
        img = Image.fromarray(img, "RGB")
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)  # resize the center crop from [crop, crop] to [width, height]

    return np.array(img).astype(np.uint8)


def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.0)
    v.clip_(0.0, 1.0)
    return v


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def split(x, z_shape=(4, 64, 64), clip_img_dim=512):
    C, H, W = z_shape
    z_dim = C * H * W
    z, clip_img = x.split([z_dim, clip_img_dim], axis=1)
    z = einops.rearrange(z, "B (C H W) -> B C H W", C=C, H=H, W=W)
    clip_img = einops.rearrange(clip_img, "B (L D) -> B L D", L=1, D=clip_img_dim)
    return z, clip_img


def combine(z, clip_img):
    z = einops.rearrange(z, "B C H W -> B (C H W)")
    clip_img = einops.rearrange(clip_img, "B L D -> B (L D)")
    return paddle.concat([z, clip_img], axis=-1)


def split_joint(x):
    z_shape = (4, 64, 64)
    clip_img_dim = 512
    text_dim = 64

    C, H, W = z_shape
    z_dim = C * H * W
    z, clip_img, text = x.split([z_dim, clip_img_dim, 77 * text_dim], axis=1)
    z = einops.rearrange(z, "B (C H W) -> B C H W", C=C, H=H, W=W)
    clip_img = einops.rearrange(clip_img, "B (L D) -> B L D", L=1, D=clip_img_dim)
    text = einops.rearrange(text, "B (L D) -> B L D", L=77, D=text_dim)
    return z, clip_img, text


def combine_joint(z, clip_img, text):
    z = einops.rearrange(z, "B C H W -> B (C H W)")
    clip_img = einops.rearrange(clip_img, "B L D -> B (L D)")
    text = einops.rearrange(text, "B L D -> B (L D)")
    return paddle.concat([z, clip_img, text], axis=-1)
