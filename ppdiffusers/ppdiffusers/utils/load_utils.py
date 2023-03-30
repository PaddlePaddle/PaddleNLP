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

import io
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import numpy as np

from .import_utils import (
    is_paddle_available,
    is_safetensors_available,
    is_torch_available,
)
from .logging import get_logger

logger = get_logger(__name__)

__all__ = ["smart_load", "torch_load", "safetensors_load"]


paddle_suffix = [".pdparams", ".pd"]
torch_suffix = [".pt", ".pth", ".bin", ".ckpt"]
safetensors_suffix = [".safetensors"]

if is_safetensors_available():
    # patch_bf16 safe tensors
    import safetensors.numpy

    np.bfloat16 = np.uint16
    np.bool = bool
    safetensors.numpy._TYPES.update({"BF16": np.uint16})

if is_torch_available():
    import torch

    # patch torch.uint16
    torch.uint16 = torch.bfloat16

if is_paddle_available():
    import paddle


MZ_ZIP_LOCAL_DIR_HEADER_SIZE = 30


def read_prefix_key(path):
    file_size = os.stat(path).st_size
    with open(path, "rb") as file_handler:
        end_index = seek_by_string(file_handler, "data.pkl", file_size)
        file_handler.seek(MZ_ZIP_LOCAL_DIR_HEADER_SIZE)
        prefix_key = file_handler.read(end_index - MZ_ZIP_LOCAL_DIR_HEADER_SIZE - len("/data.pkl"))
    return prefix_key.decode("latin")


def seek_by_string(file_handler, string: str, file_size: int) -> int:
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
        raise Exception(f"can't find the find the target string<{string}> in the file")
    return file_handler.tell()


def _maybe_decode_ascii(bytes_str: Union[bytes, str]) -> str:
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
        "BFloat16Storage": np.uint16,
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
            if name == "_rebuild_parameter":
                return _rebuild_parameter
            if name == "_rebuild_parameter_with_state":
                return _rebuild_parameter_with_state
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


def _rebuild_parameter(data, requires_grad, backward_hooks):
    return data


def _rebuild_parameter_with_state(data, requires_grad, backward_hooks, state):
    return data


def dumpy(*args, **kwarsg):
    return None


def torch_load(path: str, **pickle_load_args):
    if is_torch_available():
        import torch

        state_dict = torch.load(path, map_location="cpu")
    else:
        pickle_load_args.update({"encoding": "utf-8"})

        prefix_key = read_prefix_key(path)

        torch_zip = ZipFile(path, "r")
        loaded_storages = {}

        def load_tensor(dtype, numel, key, location):
            name = f"{prefix_key}/data/{key}"
            typed_storage = np.frombuffer(torch_zip.open(name).read()[:numel], dtype=dtype)
            return typed_storage

        def persistent_load(saved_id):
            assert isinstance(saved_id, tuple)
            typename = _maybe_decode_ascii(saved_id[0])
            data = saved_id[1:]

            assert (
                typename == "storage"
            ), f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
            storage_type, key, location, numel = data
            dtype = storage_type.dtype

            if key in loaded_storages:
                typed_storage = loaded_storages[key]
            else:
                nbytes = numel * _element_size(dtype)
                typed_storage = load_tensor(dtype, nbytes, key, _maybe_decode_ascii(location))
                loaded_storages[key] = typed_storage

            return typed_storage

        data_iostream = torch_zip.open(f"{prefix_key}/data.pkl").read()
        unpickler_stage = UnpicklerWrapperStage(io.BytesIO(data_iostream), **pickle_load_args)
        unpickler_stage.persistent_load = persistent_load
        state_dict = unpickler_stage.load()
        torch_zip.close()
    return state_dict


def convert_to_paddle(state_dict, return_numpy=False, return_global_step=False):
    pd_state_dict = {}
    # maybe we will use global_step
    if return_global_step:
        pd_state_dict["global_step"] = state_dict.pop("global_step", -1)
    state_dict = state_dict.get("state_dict", state_dict)

    # ugly
    # {
    #    "state_dict" : {"state_dict": {}, "epoch": {}, "xxxxx"}
    # }
    if "global_step" in state_dict and "state_dict" in state_dict:
        if return_global_step:
            pd_state_dict["global_step"] = state_dict.pop("global_step", -1)
        state_dict = state_dict.get("state_dict", state_dict)

    for k, v in state_dict.items():
        # maybe position id is bfloat32
        # if "position_id" in k and "int" not in str(v.dtype):
        #     v = v.numpy().astype("int64") if hasattr(v, "numpy") else v.astype("int64")
        if v.ndim == 0:
            v = v.reshape((1,))
        if not return_numpy:
            # support bfloat16
            if "torch.bfloat16" in str(v.dtype):
                v = v.float()
                pd_state_dict[k] = (
                    paddle.to_tensor(v.numpy()).cast(paddle.bfloat16)
                    if hasattr(v, "numpy")
                    else paddle.to_tensor(v).cast(paddle.bfloat16)
                )
            else:
                pd_state_dict[k] = paddle.to_tensor(v.numpy()) if hasattr(v, "numpy") else paddle.to_tensor(v)
        else:
            pd_state_dict[k] = v.numpy() if hasattr(v, "numpy") else v

    return pd_state_dict


def convert_to_numpy(state_dict):
    state_dict = state_dict.get("state_dict", state_dict)
    pd_state_dict = {}
    for k, v in state_dict.items():
        # maybe position id is bfloat32
        # if "position_id" in k and "int" not in str(v.dtype):
        #     v = v.numpy().astype("int64") if hasattr(v, "numpy") else v.astype("int64")
        if v.ndim == 0:
            v = v.reshape((1,))
    return pd_state_dict


def safetensors_load(path: str):
    if is_safetensors_available():
        try:
            if is_torch_available():
                from safetensors.torch import load_file

                data = load_file(path)
            else:
                from safetensors.numpy import load_file

                data = load_file(path)
        except Exception:
            from safetensors.numpy import load_file

            data = load_file(path)
    else:
        raise ImportError("`safetensors_load` requires the `safetensors library: `pip install safetensors`.")

    return data


def smart_load(path: str, map_location: str = "cpu", return_numpy=False, return_global_step=False):
    suffix = Path(path).suffix
    name = Path(path).name
    state_dict = None
    with paddle.device_scope(map_location):
        if suffix in paddle_suffix:
            state_dict = paddle.load(path, return_numpy=return_numpy)
            return state_dict

        if suffix in torch_suffix:
            state_dict = convert_to_paddle(torch_load(path), return_numpy, return_global_step)
            return state_dict

        if suffix in safetensors_suffix:
            state_dict = convert_to_paddle(safetensors_load(path), return_numpy, return_global_step)
            return state_dict

        # must use safetensors_load first
        try:
            state_dict = convert_to_paddle(safetensors_load(path), return_numpy, return_global_step)
            return state_dict
        except Exception:
            logger.info(f"Cant load file {name} with safetensors!")
        try:
            state_dict = convert_to_paddle(torch_load(path), return_numpy, return_global_step)
            return state_dict
        except Exception:
            logger.info(f"Cant load file {name} with torch! We will try to load this with safetensors!")
        try:
            state_dict = paddle.load(path, return_numpy=return_numpy)
            return state_dict
        except Exception:
            logger.info(f"Cant load file {name} with paddle! We will try to load this with torch/safetensors!")
    if state_dict is None:
        raise ValueError(f"Cant load {name}, currently we only support ['torch', 'safetensors', 'paddle']!")
