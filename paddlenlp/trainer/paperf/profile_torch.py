# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved
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

import torch
import contextlib


_PROFILER_ENABLED = False
_DEBUG_INFO = None


def _forward_pre_hook(module, inputs):
    global _DEBUG_INFO
    if _DEBUG_INFO:
        print(f"-- [Enter {module.__class__.__name__} forward] ")

    if _PROFILER_ENABLED:
        torch.cuda.nvtx.range_push(module.__class__.__name__ + "_fwd")
    return None


def _forward_post_hook(module, inputs, outputs):
    global _DEBUG_INFO
    if _DEBUG_INFO:
        print(f"-- [Leave {module.__class__.__name__} forward] ")

    if _PROFILER_ENABLED:
        torch.cuda.nvtx.range_pop()


def _backward_pre_hook(module, grad_output):
    global _DEBUG_INFO
    if _DEBUG_INFO:
        print(f"-- [Enter {module.__class__.__name__} backward] ")

    if _PROFILER_ENABLED:
        torch.cuda.nvtx.range_push(module.__class__.__name__ + "_bwd")
    return None


def _backward_post_hook(module, grad_input, grad_output):
    global _DEBUG_INFO
    if _DEBUG_INFO:
        print(f"-- [Leave {module.__class__.__name__} backward] ")

    if _PROFILER_ENABLED:
        torch.cuda.nvtx.range_pop()
    return None


def _register_forward_hook_recursively(module, pre_hook, post_hook):
    if not isinstance(module, torch.nn.Module):
        return

    for submodule in module.children():
        _register_forward_hook_recursively(submodule, pre_hook, post_hook)

    if pre_hook is not None:
        module.register_forward_pre_hook(hook=pre_hook)
    if post_hook is not None:
        module.register_forward_hook(hook=post_hook)


def _register_backward_hook_recursively(module, pre_hook, post_hook):
    if not isinstance(module, torch.nn.Module):
        return

    if pre_hook is not None:
        module.register_full_backward_pre_hook(hook=pre_hook)

    for submodule in module.children():
        _register_backward_hook_recursively(submodule, pre_hook, post_hook)

    if post_hook is not None:
        module.register_full_backward_hook(hook=post_hook)


def register_profile_hook(model, backward=False, debug=None):
    if debug is not None:
        global _DEBUG_INFO
        _DEBUG_INFO = debug

    if isinstance(model, torch.nn.Module):
        _register_forward_hook_recursively(model, _forward_pre_hook, _forward_post_hook)
    elif isinstance(model, list):
        for module in model:
            _register_forward_hook_recursively(
                module, _forward_pre_hook, _forward_post_hook
            )

    if debug is None and backward:
        # backwrad hook cannot be used for profile.
        if isinstance(model, torch.nn.Module):
            _register_backward_hook_recursively(
                model, _backward_pre_hook, _backward_post_hook
            )
        elif isinstance(model, list):
            for module in model:
                _register_backward_hook_recursively(
                    module, _backward_pre_hook, _backward_post_hook
                )


def _enter_emit_nvtx(record_shapes=False):
    # following code is changed from torch.autograd.profiler.emit_nvtx class.
    try:
        from torch.autograd.profiler import _run_on_profiler_start

        # https://github.com/pytorch/pytorch/blob/38d9bb5abcc31ba97927a5399b88afe2cf60bf64/torch/autograd/profiler.py#L743
        _run_on_profiler_start()
    except ImportError:
        # For older version of torch, there is no definition and call of _run_on_profiler_start
        # https://github.com/pytorch/pytorch/blob/c263bd43e8e8502d4726643bc6fd046f0130ac0e/torch/autograd/profiler.py#L619
        pass

    torch.autograd.profiler._enable_profiler(
        torch.autograd.profiler.ProfilerConfig(
            torch.autograd.profiler.ProfilerState.NVTX,
            record_shapes,
            False,
            False,
            False,
            False,
            torch.autograd.profiler._ExperimentalConfig(),
        ),
        set(),
    )


def _exit_emit_nvtx():
    torch.autograd.profiler._disable_profiler()

    try:
        from torch.autograd.profiler import _run_on_profiler_stop

        _run_on_profiler_stop()
    except ImportError:
        # For older version of torch, there is no definition and call of _run_on_profiler_stop
        pass


def switch_profile(
    iter_id, start, end, event_name=None, enable_aten_event=False, record_shapes=False
):
    global _PROFILER_ENABLED
    if event_name is None:
        event_name = "iter_{}".format(iter_id)
    if iter_id == start:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        _PROFILER_ENABLED = True
        if enable_aten_event:
            _enter_emit_nvtx(record_shapes=record_shapes)
        torch.cuda.nvtx.range_push(event_name)
    elif iter_id == end:
        torch.cuda.nvtx.range_pop()
        _PROFILER_ENABLED = False
        torch.cuda.synchronize()
        if enable_aten_event:
            _exit_emit_nvtx()
        torch.cuda.cudart().cudaProfilerStop()
    elif iter_id > start and iter_id < end:
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push(event_name)


@contextlib.contextmanager
def add_record_event(event_name):
    if _PROFILER_ENABLED:
        torch.cuda.nvtx.range_push(event_name)
        yield
        torch.cuda.nvtx.range_pop()
    else:
        yield


def push_record_event(event_name):
    if _PROFILER_ENABLED:
        torch.cuda.nvtx.range_push(event_name)


def pop_record_event():
    if _PROFILER_ENABLED:
        torch.cuda.nvtx.range_pop()
