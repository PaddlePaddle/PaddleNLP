# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 DeepSeek
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

try:
    import paddle.distributed.communication.deep_ep as deep_ep

    HAVE_DEEP_EP = True
except ImportError:
    HAVE_DEEP_EP = False

import paddle
from paddle.autograd import PyLayer
from paddle.distributed.communication.group import Group

import paddlenlp.transformers.deepep_timer as timer 
#from paperf import profile_paddle

import numpy as np


_buffer = None
_enable_dump_tensor = False
_dump_or_md5sum_check_idx = 0
_enable_md5sum_check = False

_enable_cuda_timestamp = False
_dispatch_forward_timestamp_tensor_list = []

_is_training_step_start = False


def set_is_training_step_start(is_training_step_start):
    global _is_training_step_start
    _is_training_step_start = is_training_step_start


def set_enable_cuda_timestamp(enable=True):
    global _enable_cuda_timestamp
    _enable_cuda_timestamp = enable


def gather_and_dump_timestamps():
    global _enable_cuda_timestamp
    if not _enable_cuda_timestamp:
        return

    global _dispatch_forward_timestamp_tensor_list
    dispatch_forward_timestamps = paddle.stack(_dispatch_forward_timestamp_tensor_list)
    dispatch_forward_timestamp_all_ranks = []
    paddle.distributed.all_gather(dispatch_forward_timestamp_all_ranks, dispatch_forward_timestamps)
    out = paddle.stack(dispatch_forward_timestamp_all_ranks).numpy()

    rank = paddle.distributed.get_rank()
    if rank == 0:
        np.save("dispatch-forward.npy", out)


def set_enable_dump_tensor(enable_dump=True):
    global _enable_dump_tensor
    global _dump_or_md5sum_check_idx
    _enable_dump_tensor = enable_dump
    _dump_or_md5sum_check_idx = 0


def set_enable_md5sum_check(enable_md5sum_check=True):
    global _enable_md5sum_check
    global _dump_or_md5sum_check_idx
    _enable_md5sum_check = False # enable_md5sum_check
    _dump_or_md5sum_check_idx = 0


def add_dump_or_md5sum_check_idx():
    global _enable_dump_tensor
    global _enable_md5sum_check
    if _enable_dump_tensor or _enable_md5sum_check:
        global _dump_or_md5sum_check_idx
        _dump_or_md5sum_check_idx += 1


def dump_tensor(x, name):
    global _enable_dump_tensor
    if not _enable_dump_tensor:
        return

    assert isinstance(x, paddle.Tensor)

    rank = paddle.distributed.get_rank()
    dump_dir = "/root/paddlejob/workspace/env_run/liuyiqun/outputs"
    name = dump_dir + "/" + name
    
    if x.dtype == paddle.float32 or x.dtype == paddle.int32 or x.dtype == paddle.int64 or x.dtype == paddle.bool:
        y = x.numpy()
    elif x.dtype == paddle.bfloat16:
        #y = x.view('uint16').numpy()
        y = paddle.cast(x, paddle.float32).numpy()
    else:
        assert False, f'{name}: {x.dtype} {x}'
    np.save(f"{name}_rank{rank}.npy",y)


def gather_md5sum_and_print(x, name, gather=False, exit=False):
    global _enable_md5sum_check
    global _dump_or_md5sum_check_idx
    if not _enable_md5sum_check:
        return
        
    paddle.device.synchronize()
    if gather:
        x_md5sum = paddle.to_tensor([ord(x) for x in x._md5sum()], dtype='int64')
        x_md5sum_tensor_list = []
        paddle.device.synchronize()
        paddle.distributed.all_gather(x_md5sum_tensor_list, x_md5sum)
        paddle.device.synchronize()
        x_md5sum_tensor_list = [ 
            ''.join(chr(x) for x in tensor)
            for tensor in x_md5sum_tensor_list
        ]   
        print(f"-- name: {name}_{_dump_or_md5sum_check_idx}, shape: {x.shape}, gathered md5sum: {x_md5sum_tensor_list}")
    else:
        print(f"-- name: {name}_{_dump_or_md5sum_check_idx}, shape: {x.shape}, md5sum: {x._md5sum()}")
    if exit:
        sys.exit()


def get_hidden_bytes(x: paddle.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (paddle.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.shape[1] * max(x.element_size(), 2)


def get_buffer(group: Group, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (paddle.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        deep_ep.Buffer.get_dispatch_config(group.world_size),
        deep_ep.Buffer.get_combine_config(group.world_size),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.world_size), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.world_size), num_rdma_bytes)

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        print(f"-- group.world_size: {group.world_size}, num_nvl_bytes: {num_nvl_bytes}, num_rdma_bytes: {num_rdma_bytes}")
        _buffer = deep_ep.Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


def fused_dispatch_forward_func(
    x,
    token_indices,
    token_probs,
    num_experts,
    group,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
):
    """Forward pass of fused dispatch."""

    if _is_training_step_start:
        paddle.distributed.barrier(group)
        paddle.device.synchronize()

    # Calculate layout before actual dispatch
    if isinstance(x, tuple):
        #timer_name_suffix = "_tuple"
        buffer = get_buffer(group, get_hidden_bytes(x[0]))
    else:
        #timer_name_suffix = ""
        buffer = get_buffer(group, get_hidden_bytes(x))

    #print(f"token_indices.shape: {token_indices.shape}")
    #gather_md5sum_and_print(token_indices, name="token_indices", gather=True)
    #add_dump_or_md5sum_check_idx()

    #ep_timer = timer.get_ep_timer(buffer, False)
    #ep_timer.start(f"dispatch_forward-get_dispatch_layout{timer_name_suffix}")
    #profile_paddle.push_record_event(f"dispatch_forward-get_dispatch_layout{timer_name_suffix}")
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        previous_event_,
    ) = buffer.get_dispatch_layout(
        token_indices,
        num_experts,
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
    #profile_paddle.pop_record_event()
    #ep_timer.stop(f"dispatch_forward-get_dispatch_layout{timer_name_suffix}")

    assert token_probs.dtype == paddle.float32

    # Do MoE dispatch
    # NOTES: the CPU will wait for GPU's signal to arrive,
    # so this is not compatible with CUDA graph
    #ep_timer.start(f"dispatch_forward-dispatch{timer_name_suffix}")
    #profile_paddle.push_record_event(f"dispatch_forward-dispatch{timer_name_suffix}")
    (recv_x, recv_token_indices, recv_token_probs, num_recv_tokens_per_expert_list, handle, event, time_stamp) = buffer.dispatch(
        x,
        topk_idx=token_indices,
        topk_weights=token_probs,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
    #profile_paddle.pop_record_event()
    #ep_timer.stop(f"dispatch_forward-dispatch{timer_name_suffix}")

    if _enable_cuda_timestamp:
        _dispatch_forward_timestamp_tensor_list.append(time_stamp)

    states = dict()
    states["dispatched_indices"] = recv_token_indices
    states["tokens_per_expert"] = num_recv_tokens_per_expert_list
    states["handle"] = handle

    return recv_x, recv_token_probs, states, event


def fused_dispatch_backward_func(
    grad_output,
    grad_token_probs,
    group,
    handle,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
):
    """Backward pass of fused dispatch."""
    buffer = get_buffer(group, get_hidden_bytes(grad_output))

    #ep_timer = timer.get_ep_timer(buffer, False)
    #ep_timer.start("dispatch_backward-combine")
    #profile_paddle.push_record_event("dispatch_backward-combine")
    grad_x, grad_token_probs, event = buffer.combine(
        grad_output.contiguous(),
        handle,
        topk_weights=grad_token_probs.cast(paddle.float32),
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
    #profile_paddle.pop_record_event()
    #ep_timer.stop("dispatch_backward-combine")
    return grad_x, None, grad_token_probs


def fused_combine_forward_func(
    x, group, states, previous_event=None, async_finish=False, allocate_on_comm_stream=False
):
    """Forward pass of fused combine."""
    handle = states["handle"]
    buffer = get_buffer(group, get_hidden_bytes(x))

    #ep_timer = timer.get_ep_timer(buffer, False)
    #ep_timer.start("combine_forward-combine")
    #profile_paddle.push_record_event("combine_forward-combine")
    combined_x, _, event = buffer.combine(
        x,
        handle=handle,
        async_finish=async_finish,
        previous_event=previous_event,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
    #profile_paddle.pop_record_event()
    #ep_timer.stop("combine_forward-combine")
    return combined_x


def fused_combine_backward_func(
    grad_output, group, handle, previous_event=None, async_finish=False, allocate_on_comm_stream=False
):
    """Backward pass of fused combine."""
    if isinstance(grad_output, tuple):
        buffer = get_buffer(group, get_hidden_bytes(grad_output[0]))

        #ep_timer = timer.get_ep_timer(buffer, False)
        #ep_timer.start("combine_backward-dispatch_tuple")
        #profile_paddle.push_record_event("combine_backward-dispatch_tuple")
        grad_x, _, _, _, _, event, _ = buffer.dispatch(
            (grad_output[0].contiguous(), grad_output[1].contiguous()),
            handle=handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        #profile_paddle.pop_record_event()
        #ep_timer.stop("combine_backward-dispatch_tuple")
    else:
        buffer = get_buffer(group, get_hidden_bytes(grad_output))

        #ep_timer = timer.get_ep_timer(buffer, False)
        #ep_timer.start("combine_backward-dispatch")
        #profile_paddle.push_record_event("combine_backward-dispatch")
        grad_x, _, _, _, _, event, _ = buffer.dispatch(
            grad_output.contiguous(),
            handle=handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        #profile_paddle.pop_record_event()
        #ep_timer.stop("combine_backward-dispatch")
    return grad_x


class FusedDispatch(PyLayer):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(ctx, x, token_indices, token_probs, num_experts, group, previous_event=None):
        """Forward pass of fused dispatch."""
        recv_x, recv_token_probs, states, event = fused_dispatch_forward_func(
            x, token_indices, token_probs, num_experts, group, previous_event
        )

        ctx.group = group
        ctx.handle = states["handle"]
        ctx.event = event

        return recv_x, recv_token_probs, states

    @staticmethod
    def backward(ctx, grad_output, grad_token_probs):
        """Backward pass of fused dispatch."""
        return fused_dispatch_backward_func(grad_output, grad_token_probs, ctx.group, ctx.handle)


class FusedCombine(PyLayer):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, states, previous_event=None):
        """Forward pass of fused combine."""
        combined_x = fused_combine_forward_func(x, group, states, previous_event)

        ctx.handle = states["handle"]
        ctx.group = group
        ctx.previous_event = previous_event

        return combined_x

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of fused combine."""
        return fused_combine_backward_func(grad_output, ctx.group, ctx.handle, ctx.previous_event)


if HAVE_DEEP_EP:

    def fused_dispatch(x, token_indices, token_probs, num_experts, group: Group, previous_event=None):
        """Perform fused dispatch operation if deep_ep is available.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Number of experts
            group: Process group
            previous_event: Previous CUDA event

        Returns:
            Result of FusedDispatch
        """
        return FusedDispatch.apply(x.contiguous(), token_indices, token_probs, num_experts, group, previous_event)

    def fused_combine(x, group, handle, previous_event=None):
        """Perform fused combine operation if deep_ep is available.

        Args:
            x: Input tensor
            group: Process group
            handle: Communication handle
            previous_event: Previous CUDA event

        Returns:
            Result of FusedCombine
        """
        states = dict()
        states["handle"] = handle
        return FusedCombine.apply(x, group, states, previous_event)

else:
    fused_dispatch = None
    fused_combine = None


class DispatchNode:
    def __init__(self, name="dispatch"):
        self.name = name

    def reset_statue(self):
        self.handle = None

    def forward(
        self,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Forward pass of fused dispatch."""
        recv_x, recv_token_probs, states, event = fused_dispatch_forward_func(
            x,
            token_indices,
            token_probs,
            num_experts,
            group,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        self.group = group
        self.handle = states["handle"]
        self.event = event

        return recv_x, recv_token_probs, states

    def backward(self, grad_output, grad_token_probs, previous_event=None, async_finish=False):
        """Backward pass of fused dispatch."""
        out = fused_dispatch_backward_func(
            grad_output,
            grad_token_probs,
            self.group,
            self.handle,
            previous_event=previous_event,
            async_finish=async_finish,
        )
        self.reset_statue()
        return out


class CombineNode:
    def __init__(self, name="combine"):
        self.name = name

    def reset_statue(self):
        self.handle = None

    def forward(self, x, group, handle, previous_event=None, async_finish=False):
        """Forward pass of fused combine."""
        states = dict()
        states["handle"] = handle
        combined_x = fused_combine_forward_func(
            x, group, states, previous_event=previous_event, async_finish=async_finish
        )

        self.handle = handle
        self.group = group
        self.previous_event = previous_event

        return combined_x

    def backward(self, grad_output, previous_event=None, async_finish=False):
        """Backward pass of fused combine."""
        out = fused_combine_backward_func(
            grad_output, self.group, self.handle, previous_event=previous_event, async_finish=async_finish
        )
        self.reset_statue()
        return out
