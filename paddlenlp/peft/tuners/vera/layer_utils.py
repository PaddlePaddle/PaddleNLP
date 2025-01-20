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

import collections.abc as container_abcs
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Dict, Optional, Tuple, Union

import paddle
from paddle.base.framework import Parameter
from paddle.nn import Layer


class ParameterDict(Layer):
    r"""Holds parameters in a dictionary.

    ParameterDict can be indexed like a regular Python dictionary, but Parameters it
    contains are properly registered, and will be visible by all Layer methods.
    Other objects are treated as would be done by a regular Python dictionary

    :class:`~paddle.nn.ParameterDict` is an **ordered** dictionary.
    :meth:`~paddle.nn.ParameterDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping. On the other hand, ``OrderedDict`` or another :class:`~paddle.nn.ParameterDict`
    will preserve their ordering.

    Note that the constructor, assigning an element of the dictionary and the
    :meth:`~paddle.nn.ParameterDict.update` method will convert any :class:`~paddle.Tensor` into
    :class:`~paddle.nn.Parameter`.

    Args:
        values (iterable, optional): a mapping (dictionary) of
            (string : Any) or an iterable of key-value pairs
            of type (string, Any)

    Example::

        class MyLayer(nn.Layer):
            def __init__(self) -> None:
                super().__init__()
                self.params = nn.ParameterDict({
                        'left': nn.Parameter(paddle.randn(5, 10)),
                        'right': nn.Parameter(paddle.randn(5, 10))
                })

            def forward(self, x, choice):
                x = self.params[choice].mm(x)
                return x
    """

    def __init__(self, parameters: Any = None) -> None:
        super().__init__()
        self.name = None
        self._keys: Dict[str, None] = {}
        if parameters is not None:
            self.update(parameters)

    def _key_to_attr(self, key: str) -> str:
        if not isinstance(key, str):
            raise TypeError(
                "Index given to ParameterDict cannot be used as a key as it is "
                f"not a string (type is '{type(key).__name__}'). Open an issue on "
                "github if you need non-string keys."
            )
        else:
            # Use the key as-is so that `.named_parameters()` returns the right thing
            return key

    def __getitem__(self, key: str) -> Any:
        attr = self._key_to_attr(key)
        return getattr(self, attr)

    def __setitem__(self, key: str, value: Any) -> None:
        # Note that all other function that add an entry to the dictionary part of
        # the ParameterDict end up here. So this is the only place where we need
        # to wrap things into Parameter if needed.
        # Objects added via setattr() are not in the dictionary part and thus won't
        # call into this function.
        self._keys[key] = None
        attr = self._key_to_attr(key)
        if isinstance(value, paddle.Tensor) and not isinstance(value, Parameter):
            value = Parameter(value)
        setattr(self, attr, value)

    def __delitem__(self, key: str) -> None:
        del self._keys[key]
        attr = self._key_to_attr(key)
        delattr(self, attr)

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __reversed__(self) -> Iterator[str]:
        return reversed(list(self._keys))

    def copy(self) -> "ParameterDict":
        """Return a copy of this :class:`~paddle.nn.ParameterDict` instance."""
        # We have to use an OrderedDict because the ParameterDict constructor
        # behaves differently on plain dict vs OrderedDict
        return ParameterDict(OrderedDict((k, self[k]) for k in self._keys))

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def setdefault(self, key: str, default: Optional[Any] = None) -> Any:
        """Set the default for a key in the Parameterdict.

        If key is in the ParameterDict, return its value.
        If not, insert `key` with a parameter `default` and return `default`.
        `default` defaults to `None`.

        Args:
            key (str): key to set default for
            default (Any): the parameter set to the key
        """
        if key not in self:
            self[key] = default
        return self[key]

    def clear(self) -> None:
        """Remove all items from the ParameterDict."""
        for k in self._keys.copy():
            del self[k]

    def pop(self, key: str) -> Any:
        r"""Remove key from the ParameterDict and return its parameter.

        Args:
            key (str): key to pop from the ParameterDict
        """
        v = self[key]
        del self[key]
        return v

    def popitem(self) -> Tuple[str, Any]:
        """Remove and return the last inserted `(key, parameter)` pair from the ParameterDict."""
        k, _ = self._keys.popitem()
        # We need the key in the _keys to be able to access/del
        self._keys[k] = None
        val = self[k]
        del self[k]
        return k, val

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        r"""Return the parameter associated with key if present. Otherwise return default if provided, None if not.

        Args:
            key (str): key to get from the ParameterDict
            default (Parameter, optional): value to return if key not present
        """
        return self[key] if key in self else default

    def fromkeys(self, keys: Iterable[str], default: Optional[Any] = None) -> "ParameterDict":
        r"""Return a new ParameterDict with the keys provided.

        Args:
            keys (iterable, string): keys to make the new ParameterDict from
            default (Parameter, optional): value to set for all keys
        """
        return ParameterDict((k, default) for k in keys)

    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ParameterDict keys."""
        return self._keys.keys()

    def items(self) -> Iterable[Tuple[str, Any]]:
        r"""Return an iterable of the ParameterDict key/value pairs."""
        return ((k, self[k]) for k in self._keys)

    def values(self) -> Iterable[Any]:
        r"""Return an iterable of the ParameterDict values."""
        return (self[k] for k in self._keys)

    def update(self, parameters: Union[Mapping[str, Any], "ParameterDict"]) -> None:
        r"""Update the :class:`~paddle.nn.ParameterDict` with key-value pairs from ``parameters``, overwriting existing keys.

        .. note::
            If :attr:`parameters` is an ``OrderedDict``, a :class:`~paddle.nn.ParameterDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Args:
            parameters (iterable): a mapping (dictionary) from string to
                :class:`~paddle.nn.Parameter`, or an iterable of
                key-value pairs of type (string, :class:`~paddle.nn.Parameter`)
        """
        if not isinstance(parameters, container_abcs.Iterable):
            raise TypeError(
                "ParametersDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(parameters).__name__
            )

        if isinstance(parameters, (OrderedDict, ParameterDict)):
            for key, parameter in parameters.items():
                self[key] = parameter
        elif isinstance(parameters, container_abcs.Mapping):
            for key, parameter in sorted(parameters.items()):
                self[key] = parameter
        else:
            for j, p in enumerate(parameters):
                if not isinstance(p, container_abcs.Iterable):
                    raise TypeError(
                        "ParameterDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(p).__name__
                    )
                if not len(p) == 2:
                    raise ValueError(
                        "ParameterDict update sequence element "
                        "#" + str(j) + " has length " + str(len(p)) + "; 2 is required"
                    )
                # parameters as length-2 list too cumbersome to type, see LayerDict.update comment
                self[p[0]] = p[1]  # type: ignore[assignment]

    def extra_repr(self) -> str:
        child_lines = []
        for k, p in self.items():
            if isinstance(p, paddle.Tensor):
                size_str = "x".join(str(size) for size in p.shape)
                # if p.pae in ["cuda", paddle._C._get_privateuse1_backend_name()]:
                #  device_str = f" ({p.device})"
                # else:
                #     device_str = ""
                device_str = f" ({p.place})"

                parastr = "{} containing: [{} of size {}{}]".format(
                    "Parameter" if isinstance(p, Parameter) else "Tensor",
                    # paddle.typename(p),
                    p.dtype,
                    size_str,
                    device_str,
                )
                child_lines.append("  (" + str(k) + "): " + parastr)
            else:
                child_lines.append("  (" + str(k) + "): Object of type: " + type(p).__name__)
        tmpstr = "\n".join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError("ParameterDict should not be called.")

    def __or__(self, other: "ParameterDict") -> "ParameterDict":
        copy = self.copy()
        copy.update(other)
        return copy

    def __ror__(self, other: "ParameterDict") -> "ParameterDict":
        copy = other.copy()
        copy.update(self)
        return copy

    def __ior__(self, other: "ParameterDict"):
        self.update(other)
        return self


def get_in_features_out_features(self):
    return self.weight.shape[0], self.weight.shape[1]


def ParameterFunc(data: paddle.Tensor, requires_grad=True):
    tensor = paddle.create_parameter(
        data.shape, dtype=data.dtype, default_initializer=paddle.nn.initializer.Assign(data)
    )
    if not requires_grad:
        tensor.stop_gradient = True
    return tensor


paddle.nn.ParameterDict = ParameterDict
paddle.nn.Linear.get_in_features_out_features = get_in_features_out_features
paddle.nn.Parameter = ParameterFunc
