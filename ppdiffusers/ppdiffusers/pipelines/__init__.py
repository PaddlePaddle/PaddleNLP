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

from ..utils import is_onnx_available, is_paddle_available, is_paddlenlp_available, is_fastdeploy_available

if is_paddle_available():
    from .ddim import DDIMPipeline
    from .ddpm import DDPMPipeline
    from .latent_diffusion_uncond import LDMPipeline
    from .pndm import PNDMPipeline
    from .score_sde_ve import ScoreSdeVePipeline
    from .stochastic_karras_ve import KarrasVePipeline
else:
    from ..utils.dummy_paddle_objects import *  # noqa F403

if is_paddle_available() and is_paddlenlp_available():
    from .latent_diffusion import LDMTextToImagePipeline, LDMBertModel, LDMSuperResolutionPipeline
    from .stable_diffusion import (StableDiffusionImg2ImgPipeline,
                                   StableDiffusionInpaintPipeline,
                                   StableDiffusionPipeline,
                                   StableDiffusionInpaintPipelineLegacy,
                                   StableDiffusionPipelineAllinOne)

if is_paddlenlp_available() and is_onnx_available():
    from .stable_diffusion import (
        OnnxStableDiffusionImg2ImgPipeline,
        OnnxStableDiffusionInpaintPipeline,
        OnnxStableDiffusionPipeline,
    )

if is_paddlenlp_available() and is_fastdeploy_available():
    from .stable_diffusion import (FastDeployStableDiffusionImg2ImgPipeline,
                                   FastDeployStableDiffusionInpaintPipeline,
                                   FastDeployStableDiffusionPipeline)
