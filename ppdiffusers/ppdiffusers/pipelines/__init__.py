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

from ..utils import (
    OptionalDependencyNotAvailable,
    is_fastdeploy_available,
    is_k_diffusion_available,
    is_librosa_available,
    is_paddle_available,
    is_paddlenlp_available,
)

try:
    if not is_paddle_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_paddle_objects import *  # noqa F403
else:
    from .dance_diffusion import DanceDiffusionPipeline
    from .ddim import DDIMPipeline
    from .ddpm import DDPMPipeline
    from .latent_diffusion import LDMSuperResolutionPipeline
    from .latent_diffusion_uncond import LDMPipeline
    from .pndm import PNDMPipeline
    from .repaint import RePaintPipeline
    from .score_sde_ve import ScoreSdeVePipeline
    from .stochastic_karras_ve import KarrasVePipeline


try:
    if not (is_paddle_available() and is_librosa_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_paddle_and_librosa_objects import *  # noqa F403
else:
    from .audio_diffusion import AudioDiffusionPipeline, Mel

try:
    if not (is_paddle_available() and is_paddlenlp_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_paddle_and_paddlenlp_objects import *  # noqa F403
else:
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
    from .paint_by_example import PaintByExamplePipeline
    from .stable_diffusion import (
        CycleDiffusionPipeline,
        StableDiffusionControlNetPipeline,
        StableDiffusionDepth2ImgPipeline,
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
    from .unclip import UnCLIPPipeline
    from .versatile_diffusion import (
        VersatileDiffusionDualGuidedPipeline,
        VersatileDiffusionImageVariationPipeline,
        VersatileDiffusionPipeline,
        VersatileDiffusionTextToImagePipeline,
    )
    from .vq_diffusion import VQDiffusionPipeline

try:
    if not (is_paddle_available() and is_paddlenlp_available() and is_fastdeploy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_paddle_and_paddlenlp_and_fastdeploy_objects import *  # noqa F403
else:
    from .stable_diffusion import (
        FastDeployCycleDiffusionPipeline,
        FastDeployStableDiffusionImg2ImgPipeline,
        FastDeployStableDiffusionInpaintPipeline,
        FastDeployStableDiffusionInpaintPipelineLegacy,
        FastDeployStableDiffusionMegaPipeline,
        FastDeployStableDiffusionPipeline,
    )
try:
    if not (is_paddle_available() and is_paddlenlp_available() and is_k_diffusion_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_paddle_and_paddlenlp_and_k_diffusion_objects import *  # noqa F403
else:
    from .stable_diffusion import StableDiffusionKDiffusionPipeline
