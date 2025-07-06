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

import random

import numpy as np
import paddle
import TokenDispatcherUtils as TDU

seed = 2048
hidden_size = 7168
zipped_rows = 5800
unzipped_rows = 4788
subbatch_rows = 380
dtype = paddle.bfloat16

func = TDU.tokens_zip_unique_add

paddle.seed(seed)
np.random.seed(seed)
random.seed(seed)

zipped_origin = paddle.randn([0, hidden_size], dtype=paddle.float32)
unzipped = paddle.randn([unzipped_rows, hidden_size], dtype=dtype)
index_unzipped = random.sample(range(zipped_rows), unzipped_rows)
assert len(index_unzipped) == len(set(index_unzipped))
index_unzipped = paddle.to_tensor(index_unzipped, dtype=paddle.int64)

output_dtype = zipped_origin.dtype


md5sum = None
for use_subbatch in [False, True]:
    zipped = zipped_origin.clone()
    if use_subbatch and hasattr(TDU, "tokens_zip_unique_add_subbatch"):
        num_split = (zipped_rows + subbatch_rows - 1) // subbatch_rows
        rows = [subbatch_rows] * (num_split - 1)
        if zipped_rows % subbatch_rows == 0:
            rows.append(subbatch_rows)
        else:
            rows.append(zipped_rows % subbatch_rows)
        if zipped.shape[0] == 0:
            tmp = [paddle.zeros([r, hidden_size], dtype=output_dtype) for r in rows]
        else:
            tmp = paddle.split(zipped, rows, axis=0)
        args = [tmp, unzipped, index_unzipped, zipped_rows, subbatch_rows]
        output = TDU.tokens_zip_unique_add_subbatch(*args)
        output = paddle.concat(output, axis=0)
    else:
        args = [zipped, unzipped, index_unzipped, zipped_rows]
        output = TDU.tokens_zip_unique_add(*args)

    cur_md5sum = output._md5sum()
    if md5sum is None:
        md5sum = output._md5sum()
        print(f"MD5SUM: {md5sum}")
    else:
        assert md5sum == cur_md5sum, f"{md5sum} vs {cur_md5sum}"
