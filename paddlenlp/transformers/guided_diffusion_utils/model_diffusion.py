# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""
This code is based on
https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/script_util.py
"""
from .gaussian_diffusion import get_named_beta_schedule, SpacedDiffusion, space_timesteps, ModelVarType, ModelMeanType
from .unet import UNetModel
from .sec_diff import SecondaryDiffusionImageNet2

NUM_CLASSES = 1000


def create_unet_model(
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
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_fp16=False,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def create_secondary_model():
    model = SecondaryDiffusionImageNet2()
    return model


def create_gaussian_diffusion(
    steps=250,
    learn_sigma=True,
    sigma_small=False,
    noise_schedule="linear",
    predict_xstart=False,
    rescale_timesteps=True,
):
    # propcess steps
    timestep_respacing = f"ddim{steps}"
    steps = (1000 // steps) * steps if steps < 1000 else steps

    betas = get_named_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(ModelMeanType.EPSILON if not predict_xstart else ModelMeanType.START_X),
        model_var_type=(
            (ModelVarType.FIXED_LARGE if not sigma_small else ModelVarType.FIXED_SMALL)
            if not learn_sigma
            else ModelVarType.LEARNED_RANGE
        ),
        rescale_timesteps=rescale_timesteps,
    )
