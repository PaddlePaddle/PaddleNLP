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
from __future__ import annotations

import io
import pickle
from functools import lru_cache
from typing import Union
from zipfile import ZipFile

import numpy as np


def _maybe_decode_ascii(bytes_str: Union[bytes, str]) -> str:
    # When using encoding='bytes' in Py3, some **internal** keys stored as
    # strings in Py2 are loaded as bytes. This function decodes them with
    # ascii encoding, one that Py3 uses by default.
    #
    # NOTE: This should only be used on internal keys (e.g., `typename` and
    #       `location` in `persistent_load` below!
    if isinstance(bytes_str, bytes):
        return bytes_str.decode("ascii")
    return bytes_str


@lru_cache(maxsize=None)
def _storage_type_to_dtype_to_map():
    """convert storage type to numpy dtype"""
    return {
        "DoubleStorage": np.double,
        "FloatStorage": np.float32,
        "HalfStorage": np.half,
        "LongStorage": np.int64,
        "IntStorage": np.int32,
        "ShortStorage": np.int16,
        "CharStorage": np.int8,
        "ByteStorage": np.uint8,
        "BoolStorage": np.bool_,
        "ComplexDoubleStorage": np.cdouble,
        "ComplexFloatStorage": np.cfloat,
    }


class StorageType:
    """Temp Class for Storage Type"""

    def __init__(self, name):
        self.dtype = _storage_type_to_dtype_to_map()[name]

    def __str__(self):
        return f"StorageType(dtype={self.dtype})"


def _element_size(dtype: str) -> int:
    """
    Returns the element size for a dtype, in bytes
    """
    if dtype in [np.float16, np.float32, np.float64]:
        return np.finfo(dtype).bits >> 3
    elif dtype == np.bool_:
        return 1
    else:
        return np.iinfo(dtype).bits >> 3


class UnpicklerWrapperStage(pickle.Unpickler):
    def find_class(self, mod_name, name):
        if type(name) is str and "Storage" in name:
            try:
                return StorageType(name)
            except KeyError:
                pass

        # pure torch tensor builder
        if mod_name == "torch._utils":
            return _rebuild_tensor_stage

        # pytorch_lightning tensor builder
        if "pytorch_lightning" in mod_name:
            return dumpy
        return super().find_class(mod_name, name)


def _rebuild_tensor_stage(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    # if a tensor has shape [M, N] and stride is [1, N], it's column-wise / fortran-style
    # if a tensor has shape [M, N] and stride is [M, 1], it's row-wise / C-style
    # defautls to C-style
    if stride is not None and len(stride) > 1 and stride[0] == 1 and stride[1] > 1:
        order = "F"
    else:
        order = "C"

    return storage.reshape(size, order=order)


def dumpy(*args, **kwarsg):
    return None


def load_torch(path: str, **pickle_load_args):
    """
    load torch weight file with the following steps:
    1. load the structure of pytorch weight file
    2. read the tensor data and re-construct the state-dict
    Args:
        path: the path of pytorch weight file
        **pickle_load_args: args of pickle module
    Returns:
    """
    pickle_load_args.update({"encoding": "utf-8"})
    torch_zip = ZipFile(path, "r")
    loaded_storages = {}

    def load_tensor(dtype, numel, key, location):
        name = f"archive/data/{key}"
        typed_storage = np.frombuffer(torch_zip.open(name).read()[:numel], dtype=dtype)
        return typed_storage

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        assert typename == "storage", f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
        storage_type, key, location, numel = data
        dtype = storage_type.dtype

        if key in loaded_storages:
            typed_storage = loaded_storages[key]
        else:
            nbytes = numel * _element_size(dtype)
            typed_storage = load_tensor(dtype, nbytes, key, _maybe_decode_ascii(location))
            loaded_storages[key] = typed_storage

        return typed_storage

    data_iostream = torch_zip.open("archive/data.pkl").read()
    unpickler_stage = UnpicklerWrapperStage(io.BytesIO(data_iostream), **pickle_load_args)
    unpickler_stage.persistent_load = persistent_load
    result = unpickler_stage.load()
    torch_zip.close()
    return result
