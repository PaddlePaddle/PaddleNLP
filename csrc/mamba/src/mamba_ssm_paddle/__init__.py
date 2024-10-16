# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

__version__ = "2.2.2"

import paddle


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


if not hasattr(paddle, "masked_fill"):
    paddle.masked_fill = masked_fill
if not hasattr(paddle.Tensor, "masked_fill"):
    paddle.Tensor.masked_fill = masked_fill

if not hasattr(paddle, "is_autocast_enabled"):

    def is_autocast_enabled():
        tracer = paddle.framework._dygraph_tracer()
        return False if tracer._amp_level == paddle.core.AmpLevel.O0 else True

    paddle.is_autocast_enabled = is_autocast_enabled

from mamba_ssm_paddle.ops.selective_scan_interface import (
    mamba_inner_fn,
    selective_scan_fn,
)
