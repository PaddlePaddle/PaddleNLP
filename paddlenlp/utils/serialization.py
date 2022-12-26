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
import os
import pickle
from functools import lru_cache

import numpy as np
from _io import BufferedReader

MZ_ZIP_LOCAL_DIR_HEADER_SIZE = 30


class TensorMeta:
    """
    metadata of tensor
    """

    def __init__(self, key: str, n_bytes: int, dtype: str):
        self.key = key
        self.nbytes = n_bytes
        self.dtype = dtype
        self.size = None

    def __repr__(self):
        return f"size: {self.size} key: {self.key}, nbytes: {self.nbytes}, dtype: {self.dtype}"


class SerializationError(Exception):
    """Exception for serialization"""

    pass


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
        "BoolStorage": np.bool8,
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
    elif dtype == np.bool8:
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
        if mod_name == "pytorch_lightning":
            return dumpy
        return super().find_class(mod_name, name)


def get_data_iostream(file: str, file_name="data.pkl"):
    FILENAME = f"archive/{file_name}".encode("latin")
    padding_size_plus_fbxx = 4 + 14
    data_iostream = []
    offset = MZ_ZIP_LOCAL_DIR_HEADER_SIZE + len(FILENAME) + padding_size_plus_fbxx
    with open(file, "rb") as r:
        r.seek(offset)
        for bytes_data in io.BytesIO(r.read()):
            if b".PK" in bytes_data:
                data_iostream.append(bytes_data.split(b".PK")[0])
                data_iostream.append(b".")
                break
            data_iostream.append(bytes_data)
    out = b"".join(data_iostream)
    return out, offset + len(out)


def _rebuild_tensor_stage(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    if isinstance(storage, TensorMeta):
        storage.size = size
    return storage


def dumpy(*args, **kwarsg):
    return None


def seek_by_string(file_handler: BufferedReader, string: str, file_size: int) -> int:
    """seek the index of file-handler with target words

    Args:
        file_handler (BufferedReader): file handler
        string (str): the specific string in the file
        file_size (int): size of file

    Returns:
        int: end index of target string
    """
    word_index = 0
    word_bytes = string.encode("latin")
    empty_byte = "".encode("latin")

    while word_index < len(string) and file_handler.tell() < file_size:
        content = file_handler.read(1)
        if content == empty_byte:
            break

        if word_bytes[word_index] == content[0]:
            word_index += 1
        else:
            word_index = 0

    if file_handler.tell() >= file_size - 1:
        raise SerializationError(f"can't find the find the target string<{string}> in the file")
    return file_handler.tell()


def read_prefix_key(file_handler: BufferedReader, file_size: int):
    """read the prefix key in model weight file, eg: archive/pytorch_model

    Args:
        file_handler (BufferedReader): file handler
        fiel_size (_type_): size of file

    Returns:
        _type_: _description_
    """
    end_index = seek_by_string(file_handler, "data.pkl", file_size)
    file_handler.seek(MZ_ZIP_LOCAL_DIR_HEADER_SIZE)
    prefix_key = file_handler.read(end_index - MZ_ZIP_LOCAL_DIR_HEADER_SIZE - len("/data.pkl"))
    return prefix_key


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

    # 1. load the structure of pytorch weight file
    def persistent_load_stage1(saved_id):
        assert isinstance(saved_id, tuple)
        data = saved_id[1:]
        storage_type, key, _, numel = data
        dtype = storage_type.dtype
        n_bytes = numel * _element_size(dtype)
        return TensorMeta(key, n_bytes, dtype)

    data_iostream, pre_offset = get_data_iostream(path, file_name="data.pkl")
    # 1. read the structure of storage
    unpickler_stage1 = UnpicklerWrapperStage(io.BytesIO(data_iostream), **pickle_load_args)
    unpickler_stage1.persistent_load = persistent_load_stage1
    result_stage1 = unpickler_stage1.load()

    # 2. get the metadata of weight file
    metadata = []

    def extract_maybe_dict(result):
        if isinstance(result, dict):
            for k, v in result.items():
                extract_maybe_dict(v)
        elif isinstance(result, (list, tuple)):
            for res in result:
                extract_maybe_dict(res)
        elif isinstance(result, TensorMeta):
            metadata.append(result)

    extract_maybe_dict(result_stage1)
    metadata = sorted(metadata, key=lambda x: x.key)
    # 3. parse the tensor of pytorch weight file
    stage1_key_to_tensor = {}
    content_size = os.stat(path).st_size
    with open(path, "rb") as file_handler:
        prefix_key = read_prefix_key(file_handler, content_size).decode("latin")
        file_handler.seek(pre_offset)

        for tensor_meta in metadata:
            key = tensor_meta.key
            # eg: archive/data/1FB
            filename = f"{prefix_key}/data/{key}"
            seek_by_string(file_handler, filename, content_size)
            file_handler.seek(2, 1)

            padding_offset = np.frombuffer(file_handler.read(2)[:1], dtype=np.uint8)[0]
            file_handler.seek(padding_offset, 1)

            # save the tensor info in result to re-use memory
            stage1_key_to_tensor[key] = np.frombuffer(
                file_handler.read(tensor_meta.nbytes), dtype=tensor_meta.dtype
            ).reshape(tensor_meta.size)

    def persistent_load_stage2(saved_id):
        assert isinstance(saved_id, tuple)
        key = saved_id[2]
        return stage1_key_to_tensor[key]

    # 4. read the structure of storage
    unpickler_stage2 = UnpicklerWrapperStage(io.BytesIO(data_iostream), **pickle_load_args)
    unpickler_stage2.persistent_load = persistent_load_stage2
    result_stage2 = unpickler_stage2.load()

    return result_stage2
