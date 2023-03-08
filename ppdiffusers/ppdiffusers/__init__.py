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

from .configuration_utils import ConfigMixin
from .fastdeploy_utils import FastDeployRuntimeModel
from .ppnlp_patch_utils import *
from .utils import (
    OptionalDependencyNotAvailable,
    is_fastdeploy_available,
    is_inflect_available,
    is_k_diffusion_available,
    is_librosa_available,
    is_onnx_available,
    is_paddle_available,
    is_paddlenlp_available,
    is_scipy_available,
    is_unidecode_available,
    logging,
)
from .version import VERSION as __version__

try:
    if not is_paddle_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_paddle_objects import *  # noqa F403
else:
    from .initializer import *
    from .modeling_utils import ModelMixin
    from .models import (
        AutoencoderKL,
        ControlNetModel,
        PriorTransformer,
        Transformer2DModel,
        UNet1DModel,
        UNet2DConditionModel,
        UNet2DModel,
        VQModel,
    )
    from .optimization import (
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
        get_scheduler,
    )
    from .pipeline_utils import DiffusionPipeline
    from .pipelines import (
        DanceDiffusionPipeline,
        DDIMPipeline,
        DDPMPipeline,
        KarrasVePipeline,
        LDMPipeline,
        LDMSuperResolutionPipeline,
        PNDMPipeline,
        RePaintPipeline,
        ScoreSdeVePipeline,
    )
    from .schedulers import (
        DDIMScheduler,
        DDPMScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        HeunDiscreteScheduler,
        IPNDMScheduler,
        KarrasVeScheduler,
        KDPM2AncestralDiscreteScheduler,
        KDPM2DiscreteScheduler,
        PNDMScheduler,
        RePaintScheduler,
        SchedulerMixin,
        ScoreSdeVeScheduler,
        UnCLIPScheduler,
        VQDiffusionScheduler,
    )
    from .schedulers.preconfig import PreconfigEulerAncestralDiscreteScheduler
    from .training_utils import EMAModel

try:
    if not (is_paddle_available() and is_scipy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_paddle_and_scipy_objects import *  # noqa F403
else:
    from .schedulers import LMSDiscreteScheduler
    from .schedulers.preconfig import PreconfigLMSDiscreteScheduler

try:
    if not (is_paddle_available() and is_paddlenlp_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_paddle_and_paddlenlp_objects import *  # noqa F403
else:
    from .pipelines import (
        AltDiffusionImg2ImgPipeline,
        AltDiffusionPipeline,
        CycleDiffusionPipeline,
        LDMBertModel,
        LDMTextToImagePipeline,
        PaintByExamplePipeline,
        StableDiffusionControlNetPipeline,
        StableDiffusionDepth2ImgPipeline,
        StableDiffusionImageVariationPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInpaintPipelineLegacy,
        StableDiffusionMegaPipeline,
        StableDiffusionPipeline,
        StableDiffusionPipelineAllinOne,
        StableDiffusionPipelineSafe,
        StableDiffusionUpscalePipeline,
        UnCLIPPipeline,
        VersatileDiffusionDualGuidedPipeline,
        VersatileDiffusionImageVariationPipeline,
        VersatileDiffusionPipeline,
        VersatileDiffusionTextToImagePipeline,
        VQDiffusionPipeline,
    )

try:
    if not (is_paddle_available() and is_paddlenlp_available() and is_k_diffusion_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_paddle_and_paddlenlp_and_k_diffusion_objects import *  # noqa F403
else:
    from .pipelines import StableDiffusionKDiffusionPipeline

try:
    if not (is_paddle_available() and is_paddlenlp_available() and is_fastdeploy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_paddle_and_paddlenlp_and_fastdeploy_objects import *  # noqa F403
else:
    from .pipelines import (
        FastDeployCycleDiffusionPipeline,
        FastDeployStableDiffusionImg2ImgPipeline,
        FastDeployStableDiffusionInpaintPipeline,
        FastDeployStableDiffusionInpaintPipelineLegacy,
        FastDeployStableDiffusionMegaPipeline,
        FastDeployStableDiffusionPipeline,
    )
try:
    if not (is_paddle_available() and is_librosa_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_paddle_and_librosa_objects import *  # noqa F403
else:
    from .pipelines import AudioDiffusionPipeline, Mel
