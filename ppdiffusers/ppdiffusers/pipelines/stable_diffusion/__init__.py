# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import List, Optional, Union

import numpy as np
import PIL.Image

from ...utils import (
    BaseOutput,
    OptionalDependencyNotAvailable,
    is_fastdeploy_available,
    is_k_diffusion_available,
    is_k_diffusion_version,
    is_paddle_available,
    is_paddlenlp_available,
)


@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_content_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, or `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]


try:
    if not (is_paddle_available() and is_paddlenlp_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_paddle_and_paddlenlp_objects import *  # noqa F403
else:
    from .pipeline_cycle_diffusion import CycleDiffusionPipeline
    from .pipeline_stable_diffusion import StableDiffusionPipeline
    from .pipeline_stable_diffusion_all_in_one import StableDiffusionPipelineAllinOne
    from .pipeline_stable_diffusion_attend_and_excite import (
        StableDiffusionAttendAndExcitePipeline,
    )
    from .pipeline_stable_diffusion_controlnet import StableDiffusionControlNetPipeline
    from .pipeline_stable_diffusion_depth2img import StableDiffusionDepth2ImgPipeline
    from .pipeline_stable_diffusion_image_variation import (
        StableDiffusionImageVariationPipeline,
    )
    from .pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
    from .pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
    from .pipeline_stable_diffusion_inpaint_legacy import (
        StableDiffusionInpaintPipelineLegacy,
    )
    from .pipeline_stable_diffusion_instruct_pix2pix import (
        StableDiffusionInstructPix2PixPipeline,
    )
    from .pipeline_stable_diffusion_k_diffusion import StableDiffusionKDiffusionPipeline
    from .pipeline_stable_diffusion_latent_upscale import (
        StableDiffusionLatentUpscalePipeline,
    )
    from .pipeline_stable_diffusion_mega import StableDiffusionMegaPipeline
    from .pipeline_stable_diffusion_panorama import StableDiffusionPanoramaPipeline
    from .pipeline_stable_diffusion_pix2pix_zero import (
        StableDiffusionPix2PixZeroPipeline,
    )
    from .pipeline_stable_diffusion_sag import StableDiffusionSAGPipeline
    from .pipeline_stable_diffusion_upscale import StableDiffusionUpscalePipeline
    from .pipeline_stable_unclip import StableUnCLIPPipeline
    from .pipeline_stable_unclip_img2img import StableUnCLIPImg2ImgPipeline
    from .safety_checker import StableDiffusionSafetyChecker
    from .stable_unclip_image_normalizer import StableUnCLIPImageNormalizer


try:
    if not (is_paddle_available() and is_fastdeploy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_fastdeploy_objects import *  # noqa F403
else:
    from .pipeline_fastdeploy_cycle_diffusion import FastDeployCycleDiffusionPipeline
    from .pipeline_fastdeploy_stable_diffusion import FastDeployStableDiffusionPipeline
    from .pipeline_fastdeploy_stable_diffusion_controlnet import (
        FastDeployStableDiffusionControlNetPipeline,
    )
    from .pipeline_fastdeploy_stable_diffusion_img2img import (
        FastDeployStableDiffusionImg2ImgPipeline,
    )
    from .pipeline_fastdeploy_stable_diffusion_inpaint import (
        FastDeployStableDiffusionInpaintPipeline,
    )
    from .pipeline_fastdeploy_stable_diffusion_inpaint_legacy import (
        FastDeployStableDiffusionInpaintPipelineLegacy,
    )
    from .pipeline_fastdeploy_stable_diffusion_mega import (
        FastDeployStableDiffusionMegaPipeline,
    )
