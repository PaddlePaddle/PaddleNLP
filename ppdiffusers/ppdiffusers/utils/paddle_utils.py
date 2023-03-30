# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""
Paddle utilities: Utilities related to Paddle
"""
import contextlib
import time
from typing import List, Optional, Tuple, Union

from .import_utils import is_paddle_available
from .logging import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name

if is_paddle_available():
    import paddle

    class RNGStatesTracker:
        def __init__(self):
            self.states_ = {}

        def reset(self):
            self.states_ = {}

        def remove(self, generator_name=None):
            if generator_name is not None:
                del self.states_[generator_name]

        def manual_seed(self, seed, generator_name=None):
            if generator_name is None:
                generator_name = str(time.time())
            if generator_name in self.states_:
                raise ValueError("state {} already exists".format(generator_name))
            orig_rng_state = paddle.get_cuda_rng_state()
            paddle.seed(seed)
            self.states_[generator_name] = paddle.get_cuda_rng_state()
            paddle.set_cuda_rng_state(orig_rng_state)
            return generator_name

        @contextlib.contextmanager
        def rng_state(self, generator_name=None):
            if generator_name is not None:
                if generator_name not in self.states_:
                    raise ValueError("state {} does not exist".format(generator_name))
                orig_cuda_rng_state = paddle.get_cuda_rng_state()
                paddle.set_cuda_rng_state(self.states_[generator_name])
                try:
                    yield
                finally:
                    self.states_[generator_name] = paddle.get_cuda_rng_state()
                    paddle.set_cuda_rng_state(orig_cuda_rng_state)
            else:
                yield

    RNG_STATE_TRACKER = RNGStatesTracker()

    def get_rng_state_tracker(*args, **kwargs):
        return RNG_STATE_TRACKER

    paddle.Generator = get_rng_state_tracker

    randn = paddle.randn
    rand = paddle.rand
    randint = paddle.randint

    def randn_pt(shape, dtype=None, name=None, **kwargs):
        generator = kwargs.get("generator", None)
        if generator is None:
            return randn(shape, dtype=dtype, name=name)
        else:
            with get_rng_state_tracker().rng_state(generator):
                return randn(shape, dtype=dtype, name=name)

    def rand_pt(shape, dtype=None, name=None, **kwargs):
        generator = kwargs.get("generator", None)
        if generator is None:
            return rand(shape, dtype=dtype, name=name)
        else:
            with get_rng_state_tracker().rng_state(generator):
                return rand(shape, dtype=dtype, name=name)

    def randint_pt(low=0, high=None, shape=[1], dtype=None, name=None, **kwargs):
        generator = kwargs.get("generator", None)
        if generator is None:
            return randint(low=low, high=high, shape=shape, dtype=dtype, name=name)
        else:
            with get_rng_state_tracker().rng_state(generator):
                return randint(low=low, high=high, shape=shape, dtype=dtype, name=name)

    def randn_like_pt(x, dtype=None, name=None, **kwargs):
        generator = kwargs.get("generator", None)
        if dtype is None:
            dtype = x.dtype
        return randn_pt(x.shape, dtype=dtype, generator=generator, name=name, **kwargs)

    paddle.randn = randn_pt
    paddle.rand = rand_pt
    paddle.randint = randint_pt
    paddle.randn_like = randn_like_pt

    def randn_tensor(
        shape: Union[Tuple, List],
        generator: Optional[Union[List["paddle.Generator"], "paddle.Generator"]] = None,
        dtype: Optional["paddle.dtype"] = None,
        *kwargs,
    ):
        """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
        passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
        will always be created on CPU.
        """
        if isinstance(generator, (list, tuple)):
            batch_size = shape[0]
            shape = (1,) + tuple(shape[1:])
            latents = [randn_pt(shape, generator=generator[i], dtype=dtype) for i in range(batch_size)]
            latents = paddle.concat(latents, axis=0)
        else:
            latents = randn_pt(shape, generator=generator, dtype=dtype)

        return latents

    def rand_tensor(
        shape: Union[Tuple, List],
        generator: Optional[Union[List["paddle.Generator"], "paddle.Generator"]] = None,
        dtype: Optional["paddle.dtype"] = None,
        *kwargs,
    ):
        """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
        passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
        will always be created on CPU.
        """
        if isinstance(generator, (list, tuple)):
            batch_size = shape[0]
            shape = (1,) + tuple(shape[1:])
            latents = [rand_pt(shape, generator=generator[i], dtype=dtype) for i in range(batch_size)]
            latents = paddle.concat(latents, axis=0)
        else:
            latents = rand_pt(shape, generator=generator, dtype=dtype)

        return latents

    def randint_tensor(
        low=0,
        high=None,
        shape: Union[Tuple, List] = [1],
        generator: Optional["paddle.Generator"] = None,
        dtype: Optional["paddle.dtype"] = None,
        *kwargs,
    ):
        """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
        passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
        will always be created on CPU.
        """
        latents = randint_pt(low=low, high=high, shape=shape, dtype=dtype, generator=generator)

        return latents
