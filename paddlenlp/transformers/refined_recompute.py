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

from __future__ import annotations

import contextlib
import copy
import inspect
import queue
import uuid
import weakref

import paddle
from paddle.base.framework import EagerParamBase
from paddle.distributed.fleet.meta_parallel.parallel_layers.random import (
    get_rng_state_tracker,
)
from paddle.distributed.fleet.recompute.recompute import switch_rng_state_tracker
from paddle.framework import _dygraph_tracer, core

_in_no_recompute = False
global_rr_queue_dict = {}
recompute_suffix = "@recompute"
_recompute_id = -1


class RefienedRecomputeQueue:
    def __init__(self) -> None:
        self.output_tensors_queue = queue.Queue()
        self.pack_tensors_queue = queue.Queue()


def set_recompute_id(value=-1):
    global _recompute_id
    _recompute_id = str(value)


def get_recompute_id():
    global _recompute_id
    return _recompute_id


@contextlib.contextmanager
def switch_recompute_id_ctx(value=-1):
    raw_recompute_id = get_recompute_id()
    set_recompute_id(value)
    yield
    set_recompute_id(raw_recompute_id)


def in_no_recompute_ctx():
    global _in_no_recompute
    return _in_no_recompute


def set_no_recompute(value=True):
    global _in_no_recompute
    _in_no_recompute = value


@contextlib.contextmanager
def switch_recompute_ctx(kwargs):
    for ts in kwargs.values():
        if paddle.is_tensor(ts) and not ts.name.endswith(recompute_suffix):
            ts.name = ts.name + recompute_suffix
    set_no_recompute(True)
    yield
    for ts in kwargs.values():
        if paddle.is_tensor(ts) and ts.name.endswith(recompute_suffix):
            ts.name = ts.name[: -len(recompute_suffix)]
    set_no_recompute(False)


def get_global_rr_queue_dict():
    global global_rr_queue_dict
    return global_rr_queue_dict


def print_global_rr_queue_info(name="pack"):
    return
    # queue_dict = get_global_rr_queue_dict()
    # print("{:<10} {:<20} {:<10}".format("Action", "Queue Name", "Queue Size"))
    # print("-" * 50)
    # for k, v in queue_dict.items():
    #     print("{:<10} {:<20} {:<10}".format(name, k, v.qsize()))
    # print("=" * 50)


class _NoopSaveInputs(paddle.autograd.PyLayer):
    """
    This layer does nothing but save all input tensors.
    This is used to prevent the gradients of the inputs being computed.
    """

    @staticmethod
    def forward(ctx, *args):
        """This function does nothing but save all input tensors."""
        tensors = [o for o in args if isinstance(o, paddle.Tensor)]
        ctx.save_for_backward(*tensors)
        return paddle.empty((0,), dtype=tensors[0].dtype)

    @staticmethod
    def backward(ctx, *args):
        """Should not be called since we don't support backward on this graph."""
        raise AssertionError("Did not expect to backward on this graph")


class NoRecomputeContext:
    def __init__(self, enable=True, save_for_bwd_keys=None):
        """
        initialize the RefinedRecomputeFunction object.
        """
        self._enable = enable
        self._save_for_bwd_keys = save_for_bwd_keys

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __call__(self, function, *args, **kwargs):
        # if not enable or no has grad, just run the original function
        if not self._enable or not _dygraph_tracer()._has_grad:
            return function(*args, **kwargs)

        input_kwargs = self.parse_to_kwargs(function, *args, **kwargs)

        if self._save_for_bwd_keys is not None:
            for key in self._save_for_bwd_keys:
                if key not in input_kwargs:
                    raise ValueError(
                        f"The key name `{key}` is not found in the input arguments."
                        " Please check your `save_for_bwd_keys`."
                    )
        else:
            self._save_for_bwd_keys = input_kwargs.keys()

        recompute_id_with_suffix = get_recompute_id()
        is_first_fwd = recompute_id_with_suffix.endswith("@first")
        recompute_id = recompute_id_with_suffix.split("@")[0]

        if is_first_fwd:
            if recompute_id not in global_rr_queue_dict:
                global_rr_queue_dict[recompute_id] = RefienedRecomputeQueue()

            with switch_recompute_ctx(input_kwargs):
                result = function(**input_kwargs)

            global_rr_queue_dict[recompute_id].output_tensors_queue.put(result)
            global_rr_queue_dict[recompute_id].pack_tensors_queue.put(0)
            print_global_rr_queue_info("first fwd")
        else:
            # is second fwd
            tensor_list = []
            for key in self._save_for_bwd_keys:
                val = input_kwargs.get(key, None)
                if val is not None and paddle.is_tensor(val):
                    tensor_list.append(val)

            tensor_offset = 0
            while global_rr_queue_dict[recompute_id].pack_tensors_queue.get() != 0:
                tensor_offset += 1
            if tensor_offset > 0 and len(tensor_list[:tensor_offset]) > 0:
                _NoopSaveInputs.apply(*tensor_list[:tensor_offset])

            result = global_rr_queue_dict[recompute_id].output_tensors_queue.get()

            if global_rr_queue_dict[recompute_id].output_tensors_queue.empty():
                global_rr_queue_dict.pop(recompute_id)

            print_global_rr_queue_info("second fwd")
        return result

    def parse_to_kwargs(self, function, *args, **kwargs):
        signature = inspect.signature(function)
        bound_arguments = signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        return bound_arguments.arguments


def share_buffer_to_tensor_or_param(inner_x):
    if hasattr(inner_x, "main_grad"):
        # donot deepcopy the `main_grad`` to save memory
        state = copy.deepcopy({k: v for k, v in inner_x.__dict__.items() if k != "main_grad"})
        tmp_tensor = EagerParamBase(shape=inner_x.shape, dtype=inner_x.dtype, name=inner_x.name + "cpy", **state)
        setattr(tmp_tensor, "main_grad", inner_x.main_grad)
        inner_x._unsafe_share_buffer_to(tmp_tensor)
    else:
        if inner_x.is_dist():
            # TODO(jeff41404): it seems better to use `tmp_tensor = core.eager.Tensor(inner_x)`,
            # but other errors will be triggered during the current period, and can be modified after resolution
            tmp_tensor = core.eager.Tensor(
                inner_x.dtype,
                inner_x.shape,
                inner_x.name + "cpy",
                core.VarDesc.VarType.LOD_TENSOR,
                inner_x.persistable,
                inner_x.process_mesh,
                inner_x.placements,
            )
        else:
            tmp_tensor = core.eager.Tensor(
                inner_x.dtype,
                inner_x.shape,
                inner_x.name + "cpy",
                core.VarDesc.VarType.LOD_TENSOR,
                inner_x.persistable,
            )
        inner_x._unsafe_share_buffer_to(tmp_tensor)
        tmp_tensor.stop_gradient = inner_x.stop_gradient
    return tmp_tensor


def _recompute_without_reentrant(function, preserve_rng_state=True, *args, **kwargs):
    """
    recompute without reentrant, that means use hook to implement the recompute function rather than re-entrant autograd.
    """

    if preserve_rng_state:
        cur_device = paddle.get_device()
        if "gpu:" in cur_device:
            fw_cuda_rng_state = paddle.get_cuda_rng_state()
        elif "cpu" in cur_device:
            fw_cuda_rng_state = paddle.get_rng_state()
        elif "xpu:" in cur_device:
            fw_cuda_rng_state = paddle.get_rng_state()
        elif cur_device.split(":")[0] in paddle.device.get_all_custom_device_type():
            fw_cuda_rng_state = paddle.get_rng_state(cur_device)
        else:
            raise RuntimeError(f"Recompute with RNG preserve is not support current device: {cur_device}.")
        fwd_cuda_rng_state_tracker = get_rng_state_tracker().get_states_tracker()
    tracer = _dygraph_tracer()
    is_fw_autocast = False if tracer._amp_level == core.AmpLevel.O0 else True
    if tracer._amp_level == core.AmpLevel.O2:
        amp_level = "O2"
    elif tracer._amp_level in (core.AmpLevel.O1, core.AmpLevel.O0):
        amp_level = "O1"

    if tracer._amp_dtype == "float16":
        amp_dtype = "float16"
    elif tracer._amp_dtype in ("bfloat16", "float32"):
        amp_dtype = "bfloat16"

    amp_white_list, amp_black_list = tracer._get_amp_op_list()

    class IntermediateHolder:
        def __init__(self, name, shape, dtype) -> None:
            self.name = name
            self.shape = shape
            self.dtype = dtype

    storage = weakref.WeakKeyDictionary()
    holder_list = []
    # generate a unique id for the recompute context
    recompute_id = str(int(uuid.uuid4()))

    def pack(x):
        # [PACK] in no recompute context or input tensor no need recompute, return the input tensor directly
        if in_no_recompute_ctx() and not x.name.endswith(recompute_suffix):
            return share_buffer_to_tensor_or_param(x)

        # remove the recompute suffix
        res = IntermediateHolder(x.name, x.shape, x.dtype)
        holder_list.append(weakref.ref(res))
        if x.name.endswith(recompute_suffix):
            global_rr_queue_dict[recompute_id].pack_tensors_queue.put(1)
        return res

    def unpack(x):
        # [UNPACK] in no recompute context or input tensor no need recompute, return the input tensor directly
        if paddle.is_tensor(x):
            return x

        unpack_counter = 0
        if len(storage) == 0:

            def inner_pack(inner_x):
                nonlocal unpack_counter
                unpack_counter += 1

                if unpack_counter - 1 >= len(holder_list):
                    raise Exception(
                        "Not supported to retrieve a tensor saved by autograd multiple times that is no need to recompute."
                        " Please check your `save_for_bwd_keys` first!"
                    )

                if holder_list[unpack_counter - 1]() is None:
                    return
                if inner_x is None:
                    storage[holder_list[unpack_counter - 1]()] = None
                    return

                storage[holder_list[unpack_counter - 1]()] = share_buffer_to_tensor_or_param(inner_x)
                return

            def inner_unpack(inner_x):
                raise Exception("An unexpected backward called on a tensor!")

            rng_cxt_manager = (
                contextlib.nullcontext()
                if not preserve_rng_state
                else switch_rng_state_tracker(fw_cuda_rng_state, fwd_cuda_rng_state_tracker)
            )
            with rng_cxt_manager:
                with paddle.set_grad_enabled(True):
                    with paddle.amp.auto_cast(
                        enable=is_fw_autocast,
                        custom_white_list=amp_white_list,
                        custom_black_list=amp_black_list,
                        level=amp_level,
                        dtype=amp_dtype,
                    ):
                        with switch_recompute_id_ctx(recompute_id + "@second"):
                            with paddle.autograd.saved_tensors_hooks(inner_pack, inner_unpack):
                                unused_outputs = function(*args, **kwargs)  # noqa: F841

        if x not in storage:
            raise Exception(
                "Not supported to retrieve a tensor saved by autograd multiple times that is no need to recompute."
            )
        tensor = storage.pop(x)
        assert x.shape == tensor.shape, (
            f"The shape:{x.shape} of the tensor saved by autograd is not "
            f"consistent with the original tensor shape:{tensor.shape}! Please check your `save_for_bwd_keys`!"
        )
        assert x.dtype == tensor.dtype, (
            f"The dtype:{x.dtype} of the tensor saved by autograd is not"
            f"consistent with the original tensor dtype:{tensor.dtype}! Please check your `save_for_bwd_keys`!"
        )
        return tensor

    with switch_recompute_id_ctx(recompute_id + "@first"):
        with paddle.autograd.saved_tensors_hooks(pack, unpack):
            outputs = function(*args, **kwargs)

    return outputs


def recompute_without_reentrant(function, *args, **kwargs):
    preserve = kwargs.pop("preserve_rng_state", True)
    use_reentrant = kwargs.pop("use_reentrant", True)
    assert use_reentrant, "recompute_without_reentrant only support use_reentrant=True!"
    return _recompute_without_reentrant(function, preserve, *args, **kwargs)


if __name__ == "__main__":
    # test flashmask_attention
    paddle.seed(2024)
    from paddle.nn.functional.flash_attention import flashmask_attention

    dtype = "float16"
    paddle.set_default_dtype(dtype)

    in_weight_shape = (32, 3 * 2 * 32)
    in_weight = paddle.create_parameter(shape=in_weight_shape, dtype=dtype, name="in_weight")
    in_weight.set_value(paddle.normal(0, 0.02, in_weight_shape))
    in_weight.main_grad = paddle.normal(0, 0.02, in_weight.shape).cast("float32")

    in_bias = paddle.create_parameter(shape=(in_weight.shape[-1],), dtype=dtype, name="in_bias", is_bias=True)
    in_bias.main_grad = paddle.normal(0, 0.02, in_bias.shape).cast("float32")

    out_weight_shape = (2 * 32, 32)
    out_weight = paddle.create_parameter(shape=out_weight_shape, dtype=dtype, name="out_weight")
    out_weight.set_value(paddle.normal(0, 0.02, out_weight_shape))
    out_weight.main_grad = paddle.normal(0, 0.02, out_weight.shape).cast("float32")

    def fwd(x, startend_row_indices, enable=True):
        with NoRecomputeContext(enable=enable) as no_recompute:
            qkv = no_recompute(paddle.nn.functional.linear, x, in_weight, bias=in_bias)

        q, k, v = paddle.chunk(qkv, 3, axis=-1)
        q = q.reshape([q.shape[0], q.shape[1], 2, q.shape[2] // 2])
        k = k.reshape([k.shape[0], k.shape[1], 2, v.shape[2] // 2])
        v = v.reshape([v.shape[0], k.shape[1], 2, v.shape[2] // 2])
        with NoRecomputeContext(enable=enable) as no_recompute:
            out = no_recompute(
                flashmask_attention,
                q,
                k,
                v,
                startend_row_indices=startend_row_indices,
                causal=True,
            )
        out = out.flatten(-2, -1)
        out = paddle.matmul(out, out_weight)
        return out

    x = paddle.normal(0, 0.02, (1, 128, 32))
    x.stop_gradient = False
    x_input = x
    startend_row_indices = paddle.randint(0, 128, (1, 2, 128, 1), dtype="int32")

    # 第一层
    o1 = recompute_without_reentrant(fwd, x, startend_row_indices, enable=True)
    # 第二层
    o2 = recompute_without_reentrant(fwd, o1, startend_row_indices, enable=True)
    # 第三层
    o3 = recompute_without_reentrant(fwd, o2, startend_row_indices, enable=True)

    o3.sum().backward()
    print(x_input.grad.mean())
    print(in_weight.grad.mean())
    print(out_weight.grad.mean())
