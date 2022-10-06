# coding=utf-8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team.
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

import paddle
from collections import UserDict
from typing import Any, Dict, Optional, Union

import numpy as np
from .tokenizer_utils_base import TensorType


class BatchFeature(UserDict):
    r"""
    Holds the feature extractor specific `__call__` methods.
    This class is derived from a python dictionary and can be used as a dictionary.
    Args:
        data (`dict`):
            Dictionary of lists/arrays/tensors returned by the __call__/pad methods ('input_values', 'attention_mask',
            etc.).
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in Paddle/Numpy Tensors at
            initialization.
    """

    def __init__(self,
                 data: Optional[Dict[str, Any]] = None,
                 tensor_type: Union[None, str, TensorType] = None):
        super().__init__(data)
        self.convert_to_tensors(tensor_type=tensor_type)

    def __getitem__(self, item: str):
        """
        If the key is a string, returns the value of the dict associated to `key` ('input_values', 'attention_mask',
        etc.).
        """
        if isinstance(item, str):
            return self.data[item]
        else:
            raise KeyError(
                "Indexing with integers is not available when using Python based feature extractors"
            )

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    def __getstate__(self):
        return {"data": self.data}

    def __setstate__(self, state):
        if "data" in state:
            self.data = state["data"]

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def convert_to_tensors(self,
                           tensor_type: Optional[Union[str,
                                                       TensorType]] = None):
        """
        Convert the inner content to tensors.
        Args:
            tensor_type (`str` or [`TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`TensorType`]. If
                `None`, no modification is done.
        """
        if tensor_type is None:
            return self

        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        # Get a function reference for the correct framework
        if tensor_type == TensorType.PADDLE:
            as_tensor = paddle.to_tensor
            is_tensor = paddle.is_tensor
        else:
            as_tensor = np.asarray
            is_tensor = lambda x: isinstance(x, np.ndarray)

        # Do the tensor conversion in batch
        for key, value in self.items():
            try:
                if not is_tensor(value):
                    tensor = as_tensor(value)

                    self[key] = tensor
            except:  # noqa E722
                if key == "overflowing_tokens":
                    raise ValueError(
                        "Unable to create tensor returning overflowing tokens of different lengths. "
                        "Please see if a fast version of this tokenizer is available to have this feature available."
                    )
                raise ValueError(
                    "Unable to create tensor, you should probably activate truncation and/or padding "
                    "with 'padding=True' 'truncation=True' to have batched tensors with the same length."
                )

        return self
