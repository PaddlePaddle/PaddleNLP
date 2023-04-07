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


import einops
import paddle


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = paddle.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=paddle.float64) ** 2
    return _betas.numpy()


def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.0)
    v.clip_(0.0, 1.0)
    return v


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
