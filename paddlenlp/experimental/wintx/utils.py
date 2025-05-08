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

import triton
import triton.language as tl


@triton.jit
def unpack_zp_bs(bzp, bs, BLOCK_SIZE_N, BLOCK_SIZE_K, pack_num, group_nums):

    bzp = bzp.permute(1, 0)
    bzp = bzp.expand_dims(axis=-1).expand_dims(axis=0)
    bzp = bzp.broadcast_to(pack_num, BLOCK_SIZE_N // pack_num, group_nums, BLOCK_SIZE_K // group_nums)
    bzp = bzp.permute(1, 0, 2, 3).reshape(BLOCK_SIZE_N, BLOCK_SIZE_K).permute(1, 0)

    bs = bs.permute(1, 0)
    bs = bs.expand_dims(axis=-1)
    bs = bs.broadcast_to(BLOCK_SIZE_N, group_nums, BLOCK_SIZE_K // group_nums).reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
    bs = bs.permute(1, 0)
    return bzp, bs


@triton.jit
def dequant(b, bs, bzp, b_shift_bits, BLOCK_SIZE_N, BLOCK_SIZE_K, pack_num, w_mask):

    b = b.permute(1, 0)
    b = b.expand_dims(axis=-1)
    b = b.broadcast_to(BLOCK_SIZE_N, BLOCK_SIZE_K // pack_num, pack_num).reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
    b = b.permute(1, 0)

    int_b = (b >> b_shift_bits) & w_mask
    b = (int_b - bzp) * bs
    return b


@triton.jit
def unpack_bs(bs, BLOCK_SIZE_N, BLOCK_SIZE_K, group_nums):
    bs = bs.permute(1, 0)
    bs = bs.expand_dims(axis=-1)
    bs = bs.broadcast_to(BLOCK_SIZE_N, group_nums, BLOCK_SIZE_K // group_nums).reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
    bs = bs.permute(1, 0)

    return bs


@triton.jit
def swizzle_tile(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    width = GROUP_SIZE_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    return pid_m, pid_n


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.jit
def linear_tile(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)
    return pid_m, pid_n


@triton.jit
def vectorize_load(offset, block_size):
    return tl.max_contiguous(tl.multiple_of(offset, block_size), block_size)


@triton.jit
def group_tile(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    # Program ID
    pid = tl.program_id(axis=0)
    # Number of program ids along the M axis
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    # Number of programs ids along the N axis
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # Number of programs in group
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # Id of the group this program is in
    group_id = pid // num_pid_in_group
    # Row-id of the first program in the group
    first_pid_m = group_id * GROUP_SIZE_M
    # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # *Within groups*, programs are ordered in a column-major order
    # Row-id of the program in the *launch grid*
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    # Col-id of the program in the *launch grid*
    pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n
