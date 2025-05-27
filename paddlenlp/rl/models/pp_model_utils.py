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
from typing import Any, Union

import paddle


def fwd_step_patch(func, output, self, *args, **kwargs):
    """
    Forward step patch function to handle gradient computation and loss recording during model training.
    If the current model is the last stage and in training mode, it will record the gradient of the output
    to the self._step_losses list. Otherwise, it will not perform any operations on the output.

    Args:
        func (Callable): The function being called, should be the forward function or any other function that needs to be executed.
        output (Tensor): The output of the model, which should be a tensor.
        self (Any): The model instance, which can be of type nn.Module or other custom model types.
        args (Tuple[Any], optional): Optional arguments passed to func, default is None.
        kwargs (Dict[str, Any], optional): Optional keyword arguments passed to func, default is None.

    Returns:
        None, no return value, directly modifies the self._step_losses attribute.
    """
    # Training patch
    if self.training and self.is_pipeline_last_stage():
        if getattr(self, "_step_losses", None):
            self._step_losses.append(output.detach())
        else:
            self._step_losses = [output.detach()]


def make_wrapper(func, pre_patch=None, post_patch=None):
    """
    Creates a wrapper function that allows executing additional operations before and after calling the original function.

    Args:
        func (function): The function to be wrapped.
        pre_patch (Optional[function], optional): The function to be executed before calling the original function, defaults to None.
            The function signature should be `pre_patch(func, None, *args, **kwargs)`.
        post_patch (Optional[function], optional): The function to be executed after calling the original function, defaults to None.
            The function signature should be `post_patch(func, output, *args, **kwargs)`, where `output` is the return value of the original function.

    Returns:
        function: The wrapped function, which has the same functionality as the original function but executes additional operations before and after the call.
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
    """
    Pads the length of tensors shaped [bs, seq_len] to [bs, max(seq_lens)].

    Args:
        inputs (list of paddle.Tensor or None): List of input tensors or None values.
        padding_value (int or float, optional): The value to pad with, defaults to 0.
        max_len (int, optional): The maximum length to pad to, if not provided it will be calculated.
        pad_len (int or list of int, optional): The length to pad each tensor by, if not provided it will be calculated.

    Returns:
        list of paddle.Tensor: List of padded input tensors.
    """
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
            axis=-1,
        )

    return inputs


def get_expected_keys(inputs: dict, keys: list[str]) -> Union[tuple, Any]:
    """
    Retrieve the expected key-value pairs from the inputs. If the key exists in the inputs,
    return the corresponding value; otherwise, return None. If there is only one key-value pair,
    convert it to a single element.

    Args:
        inputs (dict): A dictionary containing multiple key-value pairs to search for the expected ones.
        keys (list[str]): A list of keys to be searched for.

    Returns:
        Union[tuple, Any]: If there is only one key-value pair, return the single element;
        otherwise, return a tuple containing all key-value pairs. If any key does not exist, return None.
    """
    ret = tuple([inputs.get(k, None) for k in keys if k in inputs])
    if len(ret) == 1:
        ret = ret[0]
    return ret


def fwd_args_to_dict(fun):
    """
    Converts the function's arguments into a dictionary to support more argument formats in the prediction pipeline step.
    Assumes that no argument is of type inspect.Parameter.VAR_KEYWORD.

    Args:
        fun (Callable[[Any, Dict[str, Any]], Any]): The function to be converted. Its first argument is an instance of a non-pipeline model class,
            and subsequent arguments can be any format of non-pipeline model forward arguments. The return value is of any type.

    Returns:
        Callable[[Any, *Any, **Any], Any]: A new function that accepts the same arguments as the original function, but converts all non-self arguments
            into a dictionary and passes it as the second argument to the original function.
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
