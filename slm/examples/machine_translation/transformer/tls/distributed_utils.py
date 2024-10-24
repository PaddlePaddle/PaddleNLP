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

import paddle
import paddle.distributed as dist


def all_gather_tokens(data):
    """Gathers num of tokens from all nodes.
    `data` should be a tensor of num of tokens.
    """
    if dist.get_world_size() < 2:
        return data
    if not hasattr(all_gather_tokens, "_in_buffer") or all_gather_tokens._in_buffer is None:
        all_gather_tokens._in_buffer = data
        all_gather_tokens._out_buffers = []
    in_buffer = all_gather_tokens._in_buffer
    out_buffers = all_gather_tokens._out_buffers

    dist.all_gather(out_buffers, in_buffer)

    return paddle.add_n(out_buffers)
