# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import importlib
import inspect

import paddle


def fwd_step_patch(func, output, self, *args, **kwargs):
    # training patch
    if self.training and self.is_pipeline_last_stage():
        if getattr(self, "_step_losses", None):
            self._step_losses.append(output.detach())
        else:
            self._step_losses = [output.detach()]


def make_wrapper(func, pre_patch=None, post_patch=None):
    def wrapper(*args, **kwargs):
        if pre_patch is not None:
            pre_patch(func, None, *args, **kwargs)
        output = func(*args, **kwargs)
        if post_patch is not None:
            post_patch(func, output, *args, **kwargs)
        return output

    return wrapper


funcs = [(paddle.distributed.fleet.model.PipelineParallel._forward_step, fwd_step_patch)]

for func in funcs:
    fun, patch = func
    module = importlib.import_module(fun.__module__)
    cls_name = fun.__qualname__[: -len(fun.__name__) - 1]
    wrap_fun = make_wrapper(fun, post_patch=patch)
    cls_obj = getattr(module, cls_name)
    setattr(cls_obj, fun.__name__, wrap_fun)


@paddle.no_grad()
def pad_batches_inputs(inputs, padding_value=0, max_len=None, pad_len=None):
    """Pad length for tensors shaped [bs, seq_len] to [bs, max(seq_lens)]"""
    if pad_len is not None:
        pad_len = [pad_len] * len(inputs) if isinstance(pad_len, int) else pad_len
    elif max_len is None:
        # max_len = max([x.shape[-1] for x in inputs if x is not None])
        max_len = max([x.shape[-1] if isinstance(x, paddle.Tensor) else 0 for x in inputs])
        pad_len = [max_len - x.shape[-1] if isinstance(x, paddle.Tensor) else 0 for x in inputs]
    for i in range(len(inputs)):
        x = inputs[i]
        # if x is None or x.shape[-1] == max_len:
        if not isinstance(x, paddle.Tensor) or x.shape[-1] == max_len:
            continue
        inputs[i] = paddle.concat([x, paddle.full([x.shape[0], pad_len[i]], padding_value, dtype=x.dtype)], -1)
    return inputs


def get_expected_keys(inputs, keys):
    ret = tuple([inputs.get(k, None) for k in keys if k in inputs])
    if len(ret) == 1:
        ret = ret[0]
    return ret


def fwd_args_to_dict(fun):
    def _impl(self, *args, **kwargs):
        try:
            return fun(self, *args, **kwargs)
        except TypeError:
            # otherwise, inputs is any valid format of non_pipe_model forward args,
            # convert to dict, to support more args format in prediction_pipeline_step
            # assume no arg is inspect.Parameter.VAR_KEYWORD
            arg_dict = (
                inspect.signature(self._non_pipe_model_class.forward).bind(*((self,) + args), **kwargs).arguments
            )
            arg_dict.pop("self")
            return fun(self, arg_dict)

    return _impl
