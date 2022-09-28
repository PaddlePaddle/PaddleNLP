# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import functools
import inspect
import warnings

import paddle
from paddle.nn import Layer


def fn_args_to_dict(func, *args, **kwargs):
    """
    Inspect function `func` and its arguments for running, and extract a
    dict mapping between argument names and keys. 
    """
    if hasattr(inspect, 'getfullargspec'):
        (spec_args, spec_varargs, spec_varkw, spec_defaults, _, _,
         _) = inspect.getfullargspec(func)
    else:
        (spec_args, spec_varargs, spec_varkw,
         spec_defaults) = inspect.getargspec(func)
    # add positional argument values
    init_dict = dict(zip(spec_args, args))
    # add default argument values
    kwargs_dict = dict(zip(spec_args[-len(spec_defaults):],
                           spec_defaults)) if spec_defaults else {}
    for k in list(kwargs_dict.keys()):
        if k in init_dict:
            kwargs_dict.pop(k)
    kwargs_dict.update(kwargs)
    init_dict.update(kwargs_dict)
    return init_dict


def adapt_stale_fwd_patch(self, name, value):
    """
    Since there are some monkey patches for forward of PretrainedModel, such as
    model compression, we make these patches compatible with the latest forward
    method.
    """
    if name == "forward":
        # NOTE(guosheng): In dygraph to static, `layer.forward` would be patched
        # by an instance of `StaticFunction`. And use string compare to avoid to
        # import fluid.
        if type(value).__name__.endswith('StaticFunction'):
            return value
        if hasattr(inspect, 'getfullargspec'):
            (patch_spec_args, patch_spec_varargs, patch_spec_varkw,
             patch_spec_defaults, _, _, _) = inspect.getfullargspec(value)
            (spec_args, spec_varargs, spec_varkw, spec_defaults, _, _,
             _) = inspect.getfullargspec(self.forward)
        else:
            (patch_spec_args, patch_spec_varargs, patch_spec_varkw,
             patch_spec_defaults) = inspect.getargspec(value)
            (spec_args, spec_varargs, spec_varkw,
             spec_defaults) = inspect.getargspec(self.forward)
        new_args = [
            arg for arg in ('output_hidden_states', 'output_attentions',
                            'return_dict')
            if arg not in patch_spec_args and arg in spec_args
        ]

        if new_args:
            if self.__module__.startswith("paddlenlp"):
                warnings.warn(
                    f"The `forward` method of {self.__class__ if isinstance(self, Layer) else self} is patched and the patch "
                    "might be based on an old oversion which missing some "
                    f"arguments compared with the latest, such as {new_args}. "
                    "We automatically add compatibility on the patch for "
                    "these arguemnts, and maybe the patch should be updated.")
            else:
                warnings.warn(
                    f"The `forward` method of {self.__class__ if isinstance(self, Layer) else self} "
                    "is patched and the patch might be conflict with patches made "
                    f"by paddlenlp which seems have more arguments such as {new_args}. "
                    "We automatically add compatibility on the patch for "
                    "these arguemnts, and maybe the patch should be updated.")
            if isinstance(self, Layer) and inspect.isfunction(value):

                @functools.wraps(value)
                def wrap_fwd(*args, **kwargs):
                    for arg in new_args:
                        kwargs.pop(arg, None)
                    return value(self, *args, **kwargs)
            else:

                @functools.wraps(value)
                def wrap_fwd(*args, **kwargs):
                    for arg in new_args:
                        kwargs.pop(arg, None)
                    return value(*args, **kwargs)

            return wrap_fwd
    return value


class InitTrackerMeta(type(Layer)):
    """
    This metaclass wraps the `__init__` method of a class to add `init_config`
    attribute for instances of that class, and `init_config` use a dict to track
    the initial configuration. If the class has `_pre_init` or `_post_init`
    method, it would be hooked before or after `__init__` and called as
    `_pre_init(self, init_fn, init_args)` or `_post_init(self, init_fn, init_args)`.
    Since InitTrackerMeta would be used as metaclass for pretrained model classes,
    which always are Layer and `type(Layer)` is not `type`, thus use `type(Layer)`
    rather than `type` as base class for it to avoid inheritance metaclass
    conflicts.
    """

    def __init__(cls, name, bases, attrs):
        init_func = cls.__init__
        # If attrs has `__init__`, wrap it using accessable `_pre_init, _post_init`.
        # Otherwise, no need to wrap again since the super cls has been wraped.
        # TODO: remove reduplicated tracker if using super cls `__init__`
        pre_init_func = getattr(cls, '_pre_init',
                                None) if '__init__' in attrs else None
        post_init_func = getattr(cls, '_post_init',
                                 None) if '__init__' in attrs else None
        cls.__init__ = InitTrackerMeta.init_and_track_conf(
            init_func, pre_init_func, post_init_func)
        super(InitTrackerMeta, cls).__init__(name, bases, attrs)

    @staticmethod
    def init_and_track_conf(init_func, pre_init_func=None, post_init_func=None):
        """
        wraps `init_func` which is `__init__` method of a class to add `init_config`
        attribute for instances of that class.
        Args:
            init_func (callable): It should be the `__init__` method of a class.
            pre_init_func (callable, optional): If provided, it would be hooked after
                `init_func` and called as `pre_init_func(self, init_func, *init_args, **init_args)`.
                Default None.
            post_init_func (callable, optional): If provided, it would be hooked after
                `init_func` and called as `post_init_func(self, init_func, *init_args, **init_args)`.
                Default None.
        
        Returns:
            function: the wrapped function
        """

        @functools.wraps(init_func)
        def __impl__(self, *args, **kwargs):
            # registed helper by `pre_init_func`
            if pre_init_func:
                pre_init_func(self, init_func, *args, **kwargs)
            # keep full configuration
            init_func(self, *args, **kwargs)
            # registed helper by `post_init_func`
            if post_init_func:
                post_init_func(self, init_func, *args, **kwargs)
            self.init_config = kwargs
            if args:
                kwargs['init_args'] = args
            kwargs['init_class'] = self.__class__.__name__

        return __impl__

    def __setattr__(self, name, value):
        value = adapt_stale_fwd_patch(self, name, value)
        return super(InitTrackerMeta, self).__setattr__(name, value)
