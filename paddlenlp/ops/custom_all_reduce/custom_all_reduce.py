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

import atexit
import ctypes
from typing import List, Optional

import paddle
import paddle.distributed as dist
from paddle.distributed.communication.group import Group
from paddlenlp_ops import (
    all_reduce,
    dispose,
    init_custom_all_reduce,
    meta_size,
    register_buffer,
)

from paddlenlp.ops.custom_all_reduce import cuda_wrapper

try:
    meta_size()
    custom_ar = True
except Exception:
    custom_ar = False

_instances = []


class CustomAllreduce:

    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    # max_size: max supported allreduce size
    def __init__(self, group, max_size=2 * 8192 * 1024) -> None:
        """
        Args:
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self.disabled = True
        self.group = group

        if not custom_ar:
            # disable because of missing custom allreduce library
            # e.g. in a non-cuda environment
            return

        rank = dist.get_rank(group=self.group)
        self.rank = rank
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize custom allreduce for single GPU case.
            return

        if world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES:
            return

        if world_size < 2:
            return

        self.disabled = False

        # Buffers memory are owned by this Python class and passed to C++.
        # Meta data composes of two parts: meta data for synchronization and a
        # temporary buffer for storing intermediate allreduce results.
        self.meta_ptrs = self.create_shared_buffer(group, meta_size() + max_size)

        # This is a pre-registered IPC buffer. In eager mode, input tensors
        # are first copied into this buffer before allreduce is performed
        self.buffer_ptrs = self.create_shared_buffer(group, max_size)

        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = paddle.empty([8 * 1024 * 1024], dtype=paddle.uint8)
        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size
        self.full_nvlink = True
        self._ptr = init_custom_all_reduce(self.meta_ptrs, self.rank_data, rank, self.full_nvlink)
        register_buffer(self._ptr, self.buffer_ptrs)
        _instances.append(self)

    @staticmethod
    def create_shared_buffer(group: Group, size_in_bytes: int) -> List[int]:
        """
        Creates a shared buffer and returns a list of pointers
        representing the buffer on all processes in the group.
        """
        lib = cuda_wrapper.CudaRTLibrary()
        pointer = lib.cudaMalloc(size_in_bytes)
        # lib.cudaMemset(pointer, 2, size_in_bytes)
        handle = lib.cudaIpcGetMemHandle(pointer)
        rank = dist.get_rank(group=group)
        handles = []
        dist.all_gather_object(handles, handle, group=group)

        pointers: List[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer.value)  # type: ignore
            else:
                pointers.append(lib.cudaIpcOpenMemHandle(h).value)  # type: ignore

        return pointers

    @staticmethod
    def free_shared_buffer(group: Group, pointers: List[int], rank: Optional[int] = None) -> None:
        if rank is None:
            rank = dist.get_rank(group=group)
        lib = cuda_wrapper.CudaRTLibrary()
        lib.cudaFree(ctypes.c_void_p(pointers[rank]))

    def should_custom_ar(self, inp: paddle.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL.
        if self.world_size == 2 or self.full_nvlink:
            return inp_size < self.max_size
        return False

    def all_reduce(self, inp: paddle.Tensor, out: paddle.Tensor = None, registered: bool = False):
        """Performs an out-of-place all reduce.

        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if out is None:
            out = paddle.empty_like(inp)
        if registered:
            all_reduce(self._ptr, inp, out, 0, 0)
        else:
            all_reduce(self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size)
        return out

    def custom_all_reduce(self, input: paddle.Tensor) -> Optional[paddle.Tensor]:
        """The main allreduce API that provides support for cuda graph."""
        # When custom allreduce is disabled, this will be None.
        if self.disabled or not self.should_custom_ar(input):
            return None
        return self.all_reduce(input, registered=False)

    def close(self):
        if not self.disabled and self._ptr:
            dispose(self._ptr)
            self._ptr = 0
            self.free_shared_buffer(self.group, self.meta_ptrs, rank=self.rank)
            self.free_shared_buffer(self.group, self.buffer_ptrs, rank=self.rank)


def _cleanup_instances():
    for instance in _instances:
        instance.close()


atexit.register(_cleanup_instances)
