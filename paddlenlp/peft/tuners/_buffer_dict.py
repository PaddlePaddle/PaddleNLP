# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://bopaddle.org/api/_modules/bopaddle/utils/paddle.html

# TODO: To be removed once (if) https://github.com/pypaddle/pypaddle/pull/37385 lands

from __future__ import annotations

import collections
from collections import OrderedDict

import paddle
from paddle.nn import Layer


class BufferDict(Layer):
    r"""
    Holds buffers in a dictionary.

    BufferDict can be indexed like a regular Python dictionary, but buffers it contains are properly registered, and
    will be visible by all Layer methods. `paddle.nn.BufferDict` is an **ordered** dictionary that respects

    * the order of insertion, and
    * in `paddle.nn.BufferDict.update`, the order of the merged `OrderedDict` or another `paddle.nn.BufferDict` (the
      argument to `paddle.nn.BufferDict.update`).

    Note that `paddle.nn.BufferDict.update` with other unordered mapping types (e.g., Python's plain `dict`) does not
    preserve the order of the merged mapping.

    Args:
        buffers (iterable, optional):
            a mapping (dictionary) of (string : `paddle.Tensor`) or an iterable of key-value pairs of type (string,
            `paddle.Tensor`)

    ```python
    class MyLayer(nn.Layer):
        def __init__(self):
            super().__init__()
            self.buffers = nn.BufferDict({"left": paddle.randn(5, 10), "right": paddle.randn(5, 10)})

        def forward(self, x, choice):
            x = self.buffers[choice].mm(x)
            return x
    ```
    """

    def __init__(self, buffers=None, persistent: bool = False):
        r"""
        Args:
            buffers (`dict`):
                A mapping (dictionary) from string to `paddle.Tensor`, or an iterable of key-value pairs of type
                (string, `paddle.Tensor`).
        """
        super().__init__()
        if buffers is not None:
            self.update(buffers)

        self.persistent = persistent

    def __getitem__(self, key):
        return self._buffers[key]

    def __setitem__(self, key, buffer):
        self.register_buffer(key, buffer, persistable=self.persistent)

    def __delitem__(self, key):
        del self._buffers[key]

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.keys())

    def __contains__(self, key):
        return key in self._buffers

    def clear(self):
        """Remove all items from the BufferDict."""
        self._buffers.clear()

    def pop(self, key):
        r"""Remove key from the BufferDict and return its buffer.

        Args:
            key (`str`):
                Key to pop from the BufferDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self):
        r"""Return an iterable of the BufferDict keys."""
        return self._buffers.keys()

    def items(self):
        r"""Return an iterable of the BufferDict key/value pairs."""
        return self._buffers.items()

    def values(self):
        r"""Return an iterable of the BufferDict values."""
        return self._buffers.values()

    def update(self, buffers):
        r"""
        Update the `paddle.nn.BufferDict` with the key-value pairs from a mapping or an iterable, overwriting existing
        keys.

        Note:
            If `buffers` is an `OrderedDict`, a `paddle.nn.BufferDict`, or an iterable of key-value pairs, the order of
            new elements in it is preserved.

        Args:
            buffers (iterable):
                a mapping (dictionary) from string to `paddle.Tensor`, or an iterable of key-value pairs of type
                (string, `paddle.Tensor`).
        """
        if not isinstance(buffers, collections.abc.Iterable):
            raise TypeError(
                "BuffersDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(buffers).__name__
            )

        if isinstance(buffers, collections.abc.Mapping):
            if isinstance(buffers, (OrderedDict, BufferDict)):
                for key, buffer in buffers.items():
                    self[key] = buffer
            else:
                for key, buffer in sorted(buffers.items()):
                    self[key] = buffer
        else:
            for j, p in enumerate(buffers):
                if not isinstance(p, collections.abc.Iterable):
                    raise TypeError(
                        "BufferDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(p).__name__
                    )
                if not len(p) == 2:
                    raise ValueError(
                        "BufferDict update sequence element "
                        "#" + str(j) + " has length " + str(len(p)) + "; 2 is required"
                    )
                self[p[0]] = p[1]

    def extra_repr(self):
        child_lines = []
        for k, p in self._buffers.items():
            size_str = "x".join(str(size) for size in p.shape)
            device_str = "" if not p.place.is_gpu_place() else f" (GPU {p.place})"
            parastr = f"Buffer containing: [{p.dtype} of size {size_str}{device_str}]"
            child_lines.append("  (" + k + "): " + parastr)
        tmpstr = "\n".join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError("BufferDict should not be called.")
