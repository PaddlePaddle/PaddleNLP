# -*- coding: utf-8 -*-
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

"""
modified from  https://github.com/assafshocher/ResizeRight/blob/master/interp_methods.py
"""


from math import pi

try:
    import paddle
except ImportError:
    paddle = None

try:
    import numpy
except ImportError:
    numpy = None

if numpy is None and paddle is None:
    raise ImportError("Must have either Numpy or Paddle but both not found")


def set_framework_dependencies(x):
    """
    set framework dependencies
    """
    if type(x) is numpy.ndarray:

        def to_dtype(a):
            return a

        fw = numpy
    else:

        def to_dtype(a):
            return a.cast(x.dtype)

        fw = paddle
    # eps = fw.finfo(fw.float32).eps
    eps = numpy.finfo(numpy.float32).eps
    return fw, to_dtype, eps


def support_sz(sz):
    """
    support_sz wrapper
    """

    def wrapper(f):
        """wrapper"""
        f.support_sz = sz
        return f

    return wrapper


@support_sz(4)
def cubic(x):
    """cubic"""
    fw, to_dtype, eps = set_framework_dependencies(x)
    absx = fw.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1.0) * to_dtype(absx <= 1.0) + (
        -0.5 * absx3 + 2.5 * absx2 - 4.0 * absx + 2.0
    ) * to_dtype((1.0 < absx) & (absx <= 2.0))


@support_sz(4)
def lanczos2(x):
    """lanczos2"""
    fw, to_dtype, eps = set_framework_dependencies(x)
    return ((fw.sin(pi * x) * fw.sin(pi * x / 2) + eps) / ((pi**2 * x**2 / 2) + eps)) * to_dtype(abs(x) < 2)


@support_sz(6)
def lanczos3(x):
    """lanczos3"""
    fw, to_dtype, eps = set_framework_dependencies(x)
    return ((fw.sin(pi * x) * fw.sin(pi * x / 3) + eps) / ((pi**2 * x**2 / 3) + eps)) * to_dtype(abs(x) < 3)


@support_sz(2)
def linear(x):
    """linear"""
    fw, to_dtype, eps = set_framework_dependencies(x)
    return (x + 1) * to_dtype((-1 <= x) & (x < 0)) + (1 - x) * to_dtype((0 <= x) & (x <= 1))


@support_sz(1)
def box(x):
    """box"""
    fw, to_dtype, eps = set_framework_dependencies(x)
    return to_dtype((-1 <= x) & (x < 0)) + to_dtype((0 <= x) & (x <= 1))
