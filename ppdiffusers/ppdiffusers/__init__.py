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

from .ppnlp_patch_utils import *
from .utils import (
    is_fastdeploy_available,
    is_inflect_available,
    is_onnx_available,
    is_paddle_available,
    is_paddlenlp_available,
    is_scipy_available,
    is_unidecode_available,
)
from .version import VERSION

__version__ = VERSION

from .configuration_utils import ConfigMixin
from .fastdeploy_utils import FastDeployRuntimeModel
from .utils import logging

if is_paddle_available():
    from .initializer import *
    from .modeling_utils import ModelMixin
    from .models import (
        AutoencoderKL,
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
        ScoreSdeVpScheduler,
        VQDiffusionScheduler,
    )
    from .training_utils import EMAModel
else:
    from .utils.dummy_paddle_objects import *  # noqa F403

if is_paddle_available() and is_scipy_available():
    from .schedulers import LMSDiscreteScheduler
else:
    from .utils.dummy_paddle_and_scipy_objects import *  # noqa F403

if is_paddle_available() and is_paddlenlp_available():
    from .pipelines import (
        AltDiffusionImg2ImgPipeline,
        AltDiffusionPipeline,
        CycleDiffusionPipeline,
        LDMBertModel,
        LDMTextToImagePipeline,
        StableDiffusionImageVariationPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInpaintPipelineLegacy,
        StableDiffusionMegaPipeline,
        StableDiffusionPipeline,
        StableDiffusionPipelineAllinOne,
        StableDiffusionPipelineSafe,
        StableDiffusionUpscalePipeline,
        VersatileDiffusionDualGuidedPipeline,
        VersatileDiffusionImageVariationPipeline,
        VersatileDiffusionPipeline,
        VersatileDiffusionTextToImagePipeline,
        VQDiffusionPipeline,
    )
else:
    from .utils.dummy_paddle_and_paddlenlp_objects import *  # noqa F403

if is_paddle_available() and is_paddlenlp_available() and is_fastdeploy_available():
    from .pipelines import (
        FastDeployStableDiffusionImg2ImgPipeline,
        FastDeployStableDiffusionInpaintPipeline,
        FastDeployStableDiffusionInpaintPipelineLegacy,
        FastDeployStableDiffusionMegaPipeline,
        FastDeployStableDiffusionPipeline,
    )
else:
    from .utils.dummy_paddle_and_paddlenlp_and_fastdeploy_objects import *  # noqa F403
