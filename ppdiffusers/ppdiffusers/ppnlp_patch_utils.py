# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import functools, builtins, copy
from .utils import is_paddle_available, is_paddlenlp_available
from types import FunctionType, MethodType


def copy_func(f):
    "Copy a non-builtin function (NB `copy.copy` does not work for this)"
    if not isinstance(f, FunctionType): return copy.copy(f)
    fn = FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__,
                      f.__closure__)
    fn.__kwdefaults__ = f.__kwdefaults__
    fn.__dict__.update(f.__dict__)
    fn.__annotations__.update(f.__annotations__)
    fn.__qualname__ = f.__qualname__
    return fn


# copied from https://github.com/fastai/fastcore/blob/c9b4c088d3706569c076e7c197c724730be190ab/fastcore/basics.py#L938-L954
def patch_to(cls, as_prop=False, cls_method=False):
    "Decorator: add `f` to `cls`"
    if not isinstance(cls, (tuple, list)): cls = (cls, )

    def _inner(f):
        for c_ in cls:
            nf = copy_func(f)
            nm = f.__name__
            # `functools.update_wrapper` when passing patched function to `Pipeline`, so we do it manually
            for o in functools.WRAPPER_ASSIGNMENTS:
                setattr(nf, o, getattr(f, o))
            nf.__qualname__ = f"{c_.__name__}.{nm}"
            if cls_method:
                setattr(c_, nm, MethodType(nf, c_))
            else:
                setattr(c_, nm, property(nf) if as_prop else nf)
        # Avoid clobbering existing functions
        return globals().get(nm, builtins.__dict__.get(nm, None))

    return _inner


if is_paddle_available() and is_paddlenlp_available():
    import paddle
    from paddlenlp.transformers import PretrainedModel
    # NEG_INF = float("-inf")
    # use -1e4 as NEG_INF
    import paddlenlp.transformers.clip.modeling
    paddlenlp.transformers.clip.modeling.NEG_INF = -1e4

    @patch_to(PretrainedModel, as_prop=True)
    def dtype(self):
        try:
            return next(self.named_parameters())[1].dtype
        except StopIteration:
            return paddle.get_default_dtype()

    @patch_to(PretrainedModel, as_prop=True)
    def device(self):
        try:
            return next(self.named_parameters())[1].place
        except StopIteration:
            return paddle.get_device()
