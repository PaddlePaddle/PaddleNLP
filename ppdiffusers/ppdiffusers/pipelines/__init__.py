# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
# flake8: noqa

from ..utils import is_fastdeploy_available, is_paddle_available, is_paddlenlp_available

if is_paddle_available():
    from .dance_diffusion import DanceDiffusionPipeline
    from .ddim import DDIMPipeline
    from .ddpm import DDPMPipeline
    from .latent_diffusion import LDMSuperResolutionPipeline
    from .latent_diffusion_uncond import LDMPipeline
    from .pndm import PNDMPipeline
    from .repaint import RePaintPipeline
    from .score_sde_ve import ScoreSdeVePipeline
    from .stochastic_karras_ve import KarrasVePipeline
else:
    from ..utils.dummy_paddle_objects import *  # noqa F403

if is_paddle_available() and is_paddlenlp_available():
    from .alt_diffusion import (
        AltDiffusionImg2ImgPipeline,
        AltDiffusionPipeline,
        RobertaSeriesModelWithTransformation,
    )
    from .latent_diffusion import (
        LDMBertModel,
        LDMSuperResolutionPipeline,
        LDMTextToImagePipeline,
    )
    from .stable_diffusion import (
        CycleDiffusionPipeline,
        StableDiffusionImageVariationPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInpaintPipelineLegacy,
        StableDiffusionMegaPipeline,
        StableDiffusionPipeline,
        StableDiffusionPipelineAllinOne,
        StableDiffusionUpscalePipeline,
    )
    from .stable_diffusion_safe import StableDiffusionPipelineSafe
    from .versatile_diffusion import (
        VersatileDiffusionDualGuidedPipeline,
        VersatileDiffusionImageVariationPipeline,
        VersatileDiffusionPipeline,
        VersatileDiffusionTextToImagePipeline,
    )
    from .vq_diffusion import VQDiffusionPipeline

if is_paddlenlp_available() and is_fastdeploy_available():
    from .stable_diffusion import (
        FastDeployStableDiffusionImg2ImgPipeline,
        FastDeployStableDiffusionInpaintPipeline,
        FastDeployStableDiffusionInpaintPipelineLegacy,
        FastDeployStableDiffusionMegaPipeline,
        FastDeployStableDiffusionPipeline,
    )
