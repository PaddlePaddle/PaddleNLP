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
    if not (is_paddlenlp_available() and is_paddle_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_paddle_and_paddlenlp_objects import (
        UniDiffuserImageGenerationPipeline,
        UniDiffuserImageToTextPipeline,
        UniDiffuserImageVariationPipeline,
        UniDiffuserJointGenerationPipeline,
        UniDiffuserPipeline,
        UniDiffuserTextGenerationPipeline,
        UniDiffuserTextToImagePipeline,
        UniDiffuserTextVariationPipeline,
    )
else:
    from .pipeline_unidiffuser import UniDiffuserPipeline
    from .pipeline_unidiffuser_i import UniDiffuserImageGenerationPipeline
    from .pipeline_unidiffuser_i2t import UniDiffuserImageToTextPipeline
    from .pipeline_unidiffuser_i2t2i import UniDiffuserImageVariationPipeline
    from .pipeline_unidiffuser_joint import UniDiffuserJointGenerationPipeline
    from .pipeline_unidiffuser_t import UniDiffuserTextGenerationPipeline
    from .pipeline_unidiffuser_t2i import UniDiffuserTextToImagePipeline
    from .pipeline_unidiffuser_t2i2t import UniDiffuserTextVariationPipeline
