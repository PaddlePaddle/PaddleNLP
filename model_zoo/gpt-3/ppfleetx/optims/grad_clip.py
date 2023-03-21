# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved
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

import paddle
import paddle.distributed.fleet as fleet
from paddle.distributed import collective
from paddle.fluid import core, layers
from paddle.fluid.dygraph import base as imperative_base
from paddle.nn.clip import ClipGradBase, ClipGradByGlobalNorm, _squared_l2_norm
from ppfleetx.distributed.apis import env
