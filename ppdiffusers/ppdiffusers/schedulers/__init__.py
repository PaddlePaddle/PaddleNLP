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

from ..utils import is_scipy_available, is_paddle_available

if is_paddle_available():
    from .scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
    from .scheduling_ddim import DDIMScheduler
    from .scheduling_ddpm import DDPMScheduler
    from .scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
    from .scheduling_karras_ve import KarrasVeScheduler
    from .scheduling_pndm import PNDMScheduler
    from .scheduling_sde_ve import ScoreSdeVeScheduler
    from .scheduling_sde_vp import ScoreSdeVpScheduler
    from .scheduling_utils import SchedulerMixin
else:
    from ..utils.dummy_paddle_objects import *  # noqa F403

if is_scipy_available() and is_paddle_available():
    from .scheduling_lms_discrete import LMSDiscreteScheduler
else:
    from ..utils.dummy_paddle_and_scipy_objects import *  # noqa F403
