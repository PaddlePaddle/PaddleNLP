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

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL

from ...utils import BaseOutput, is_paddle_available, is_paddlenlp_available


@dataclass
# Copied from diffusers.pipelines.stable_diffusion.__init__.StableDiffusionPipelineOutput with Stable->Alt
class AltDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Alt Diffusion pipelines.

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


if is_paddlenlp_available() and is_paddle_available():
    from .modeling_roberta_series import RobertaSeriesModelWithTransformation
    from .pipeline_alt_diffusion import AltDiffusionPipeline
    from .pipeline_alt_diffusion_img2img import AltDiffusionImg2ImgPipeline
