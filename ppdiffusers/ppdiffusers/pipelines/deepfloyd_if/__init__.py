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
from typing import List, Optional, Union

import numpy as np
import PIL

from ...utils import (
    BaseOutput,
    OptionalDependencyNotAvailable,
    is_paddle_available,
    is_paddlenlp_available,
)
from .timesteps import (
    fast27_timesteps,
    smart27_timesteps,
    smart50_timesteps,
    smart100_timesteps,
    smart185_timesteps,
    super27_timesteps,
    super40_timesteps,
    super100_timesteps,
)


@dataclass
class IFPipelineOutput(BaseOutput):
    """
    Args:
    Output class for Stable Diffusion pipelines.
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content or a watermark. `None` if safety checking could not be performed.
        watermark_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely has a watermark. `None` if safety
            checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_detected: Optional[List[bool]]
    watermark_detected: Optional[List[bool]]


try:
    if not (is_paddle_available() and is_paddlenlp_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_paddle_and_paddlenlp_objects import *  # noqa F403
else:
    from .pipeline_if import IFPipeline
    from .pipeline_if_img2img import IFImg2ImgPipeline
    from .pipeline_if_img2img_superresolution import IFImg2ImgSuperResolutionPipeline
    from .pipeline_if_inpainting import IFInpaintingPipeline
    from .pipeline_if_inpainting_superresolution import (
        IFInpaintingSuperResolutionPipeline,
    )
    from .pipeline_if_superresolution import IFSuperResolutionPipeline
    from .safety_checker import IFSafetyChecker
    from .watermark import IFWatermarker
