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
# flake8: noqa

from ...utils import (
    OptionalDependencyNotAvailable,
    is_paddle_available,
    is_paddlenlp_available,
)

try:
    if not (is_paddlenlp_available() and is_paddle_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_paddle_and_paddlenlp_objects import (
        VersatileDiffusionDualGuidedPipeline,
        VersatileDiffusionImageVariationPipeline,
        VersatileDiffusionPipeline,
        VersatileDiffusionTextToImagePipeline,
    )
else:
    from .modeling_text_unet import UNetFlatConditionModel
    from .pipeline_versatile_diffusion import VersatileDiffusionPipeline
    from .pipeline_versatile_diffusion_dual_guided import (
        VersatileDiffusionDualGuidedPipeline,
    )
    from .pipeline_versatile_diffusion_image_variation import (
        VersatileDiffusionImageVariationPipeline,
    )
    from .pipeline_versatile_diffusion_text_to_image import (
        VersatileDiffusionTextToImagePipeline,
    )
