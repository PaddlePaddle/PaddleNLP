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

from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    ColumnSequenceParallelLinear,
    RowSequenceParallelLinear,
)
from paddle.nn import Linear

from paddlenlp.utils.tools import get_env_device

from .mc2_parallel_linear import MC2ColumnSeqParallelLinear, MC2RowSeqParallelLinear

if MC2ColumnSeqParallelLinear is not None and MC2RowSeqParallelLinear is not None:
    ColumnSequenceParallelLinear = MC2ColumnSeqParallelLinear  # noqa: F811
    RowSequenceParallelLinear = MC2RowSeqParallelLinear  # noqa: F811
elif get_env_device() == "xpu":
    try:
        from paddle_xpu.layers.nn import ColumnParallelLinear as XPUColumnParallelLinear
        from paddle_xpu.layers.nn import Linear as XPULinear  # noqa: F401
        from paddle_xpu.layers.nn import RowParallelLinear as XPURowParallelLinear
        from paddle_xpu.layers.nn.sequence_parallel import (  # noqa: F401
            XPUColumnSequenceParallelLinear,
            XPURowSequenceParallelLinear,
        )

        Linear = XPULinear  # noqa: F811
        ColumnParallelLinear = XPUColumnParallelLinear  # noqa: F811
        RowParallelLinear = XPURowParallelLinear  # noqa: F811
        ColumnSequenceParallelLinear = XPUColumnSequenceParallelLinear  # noqa: F811
        RowSequenceParallelLinear = XPURowSequenceParallelLinear  # noqa: F811
    except ImportError:
        # It's OK, just use paddle's Linear layers
        pass
