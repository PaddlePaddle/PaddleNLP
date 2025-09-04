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


def tokens_zip_unique_add_with_subbatch(zipped, unzipped, index_unzipped, zipped_rows, subbatch_rows=None):
    if subbatch_rows is None or subbatch_rows <= 0 or zipped_rows <= 0:
        return TDU.tokens_zip_unique_add(zipped, unzipped, index_unzipped, zipped_rows)
    else:
        if isinstance(zipped, paddle.Tensor):
            num_split = (zipped_rows + subbatch_rows - 1) // subbatch_rows
            remainder = zipped_rows % subbatch_rows
            if remainder == 0:
                rows = [subbatch_rows] * num_split
            else:
                rows = [subbatch_rows] * (num_split - 1) + [remainder]

            if zipped.shape[0] == 0:
                dtype = zipped.dtype
                hidden_size = zipped.shape[1]
                zipped = [paddle.zeros([r, hidden_size], dtype=dtype) for r in rows]
            else:
                zipped = paddle.split(zipped, rows, axis=0)
        return TDU.tokens_zip_unique_add_subbatch(zipped, unzipped, index_unzipped, zipped_rows, subbatch_rows)


def generate_index_unzipped(zipped_rows, unzipped_rows):
    index = random.sample(range(zipped_rows), unzipped_rows)
    assert len(index) == len(set(index))
    return paddle.to_tensor(index, dtype=paddle.int64)


def main():
    seed = 2048
    hidden_size = 7168
    zipped_rows = 5800
    unzipped_rows = 4788
    subbatch_rows = 380
    dtype = paddle.bfloat16

    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    zipped_origin = paddle.randn([zipped_rows, hidden_size], dtype=paddle.float32)
    unzipped = paddle.randn([unzipped_rows, hidden_size], dtype=dtype)
    index_unzipped = generate_index_unzipped(zipped_rows, unzipped_rows)

    md5sum = None
    for use_subbatch in [False, True]:
        random.seed(seed + 100)
        args = [zipped_origin.clone(), unzipped, index_unzipped, zipped_rows]
        if use_subbatch and hasattr(TDU, "tokens_zip_unique_add_subbatch"):
            args.append(subbatch_rows)

        for _ in range(4):
            output = tokens_zip_unique_add_with_subbatch(*args)
            args[0] = output
            args[2] = generate_index_unzipped(zipped_rows, unzipped_rows)

        if isinstance(output, (list, tuple)):
            output = paddle.concat(output, axis=0)

        cur_md5sum = output._md5sum()
        if md5sum is None:
            md5sum = output._md5sum()
            print(f"MD5SUM: {md5sum}")
        else:
            assert md5sum == cur_md5sum, f"{md5sum} vs {cur_md5sum}"


if __name__ == "__main__":
    main()
