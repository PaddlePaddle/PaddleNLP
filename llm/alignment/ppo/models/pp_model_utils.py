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
    """
    前向步骤补丁函数，用于处理模型在训练过程中的梯度计算和损失记录。
    如果当前模型是最后一个阶段并且正在进行训练，则会将输出的梯度记录到self._step_losses列表中。
    否则，不会对输出进行任何操作。

    Args:
        func (Callable): 被调用的函数，应该是forward函数或者其他需要执行的函数。
        output (Tensor): 模型的输出，应该是一个张量。
        self (Any): 模型实例，可以是nn.Module类型或其他自定义模型类型。
        args (Tuple[Any], optional): 传递给func的可选参数，默认为None。
        kwargs (Dict[str, Any], optional): 传递给func的可选关键字参数，默认为None。

    Returns:
        None, 无返回值，直接修改了self._step_losses属性。
    """
    # training patch
    if self.training and self.is_pipeline_last_stage():
        if getattr(self, "_step_losses", None):
            self._step_losses.append(output.detach())
        else:
            self._step_losses = [output.detach()]


def make_wrapper(func, pre_patch=None, post_patch=None):
    """
    创建一个包装函数，可以在调用原始函数前后执行额外的操作。

    Args:
        func (function): 需要被包装的函数。
        pre_patch (Optional[function], optional): 在调用原始函数前执行的函数，默认为None。
            函数签名应该是 `pre_patch(func, None, *args, **kwargs)`。
        post_patch (Optional[function], optional): 在调用原始函数后执行的函数，默认为None。
            函数签名应该是 `post_patch(func, output, *args, **kwargs)`，其中output是原始函数的返回值。

    Returns:
        function: 包装后的函数，具有与原始函数相同的功能，但会在调用前后执行额外的操作。
    """

    def wrapper(*args, **kwargs):
        if pre_patch is not None:
            pre_patch(func, None, *args, **kwargs)
        output = func(*args, **kwargs)
        if post_patch is not None:
            post_patch(func, output, *args, **kwargs)
        return output

    return wrapper


funcs = [
    (
        paddle.distributed.fleet.model.PipelineParallel._forward_step,
        fwd_step_patch,
    )
]

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
        inputs[i] = paddle.concat(
            [
                x,
                paddle.full([x.shape[0], pad_len[i]], padding_value, dtype=x.dtype),
            ],
            -1,
        )
    return inputs


def get_expected_keys(inputs, keys):
    """
    获取预期的键值对，如果输入中存在则返回该键值对，否则返回None。
    如果键值对只有一个，则将其转换为单个元素。

    Args:
        inputs (dict): 包含多个键值对的字典，用于查找预期的键值对。
        keys (list[str]): 需要查找的键列表。

    Returns:
        Union[tuple, Any]: 如果键值对只有一个，则返回单个元素；否则返回包含所有键值对的元组。如果任何键不存在，则返回None。
    """
    ret = tuple([inputs.get(k, None) for k in keys if k in inputs])
    if len(ret) == 1:
        ret = ret[0]
    return ret


def fwd_args_to_dict(fun):
    """
    将函数的参数转换为字典，用于支持更多的参数格式在预测流程步骤中。
    假设没有参数是inspect.Parameter.VAR_KEYWORD。

    Args:
        fun (Callable[[Any, Dict[str, Any]], Any]): 需要转换的函数，其第一个参数是非管道模型类实例，后续参数可以是任意格式的非管道模型前向传输参数，返回值是任意类型。

    Returns:
        Callable[[Any, *Any, **Any], Any]: 返回一个新的函数，接收与原函数相同的参数，但是将所有非self参数转换为字典形式，并作为第二个参数传入原函数。
    """

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
