# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
import TokenDispatcherUtils as TDU

subbatch_rows = 30
remainder_row = 0
left_shape = [100, 200]

for num in range(33):
    for dtype in [paddle.bfloat16, paddle.float32]:
        x = []
        for i in range(num - 1):
            x.append(paddle.randn([subbatch_rows] + left_shape, dtype=paddle.float32))
        x.append(
            paddle.randn([subbatch_rows if remainder_row == 0 else remainder_row] + left_shape, dtype=paddle.float32)
        )

        y1 = paddle.concat(x, axis=0).astype(dtype)
        y2 = TDU.merge_subbatch_cast(x, dtype)

        diff = np.abs(y1.numpy() - y2.numpy()).max()
        assert diff == 0, diff
