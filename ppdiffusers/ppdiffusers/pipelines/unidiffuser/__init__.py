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
    is_einops_available,
    is_paddle_available,
    is_paddlenlp_available,
)


@dataclass
class ImageTextPipelineOutput(BaseOutput):
    """
    Output class for UniDiffuser pipelines.
    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        prompt (`List[str]` or `str`)
            List of prompts.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    texts: Union[List[str], str]


try:
    if not (is_paddlenlp_available() and is_paddle_available() and is_einops_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_paddle_and_paddlenlp_and_einops_objects import (
        UniDiffuserPipeline,
    )
    from ...utils.dummy_paddle_and_paddlenlp_objects import CaptionDecoder
else:
    from .caption_decoder import CaptionDecoder
    from .pipeline_unidiffuser import UniDiffuserPipeline
