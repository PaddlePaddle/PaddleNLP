# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
from .log import logger
import os
import pickle
from typing import Any, Dict, Union
from contextlib import closing, contextmanager

MAGIC_NUMBER = 0x1950a86a20f9469cfc6c
PROTOCOL_VERSION = 1001
pickle_load_args = dict()
pickle_load_args['encoding'] = 'utf-8'
_package_registry = []


def _maybe_decode_ascii(bytes_str: Union[bytes, str]) -> str:
    # When using encoding='bytes' in Py3, some **internal** keys stored as
    # strings in Py2 are loaded as bytes. This function decodes them with
    # ascii encoding, one that Py3 uses by default.
    #
    # NOTE: This should only be used on internal keys (e.g., `typename` and
    #       `location` in `persistent_load` below!
    if isinstance(bytes_str, bytes):
        return bytes_str.decode('ascii')
    return bytes_str


def default_restore_location(storage, location):
    for _, _, fn in _package_registry:
        result = fn(storage, location)
        if result is not None:
            return result
    raise RuntimeError("don't know how to restore data location")


def _cpu_deserialize(obj, location):
    if location == 'cpu':
        return obj


def register_package(priority, tagger, deserializer):
    queue_elem = (priority, tagger, deserializer)
    _package_registry.append(queue_elem)
    _package_registry.sort()


register_package(10, 'cpu', _cpu_deserialize)


def persistent_load(saved_id):
    assert isinstance(saved_id, tuple)
    typename = _maybe_decode_ascii(saved_id[0])
    data = saved_id[1:]

    if typename == 'storage':
        data_type, root_key, location, size, view_metadata = data
        location = _maybe_decode_ascii(location)
        if root_key not in deserialized_objects:
            obj = data_type(size)
            obj._torch_load_uninitialized = True
            deserialized_objects[root_key] = default_restore_location(obj,
                                                                      location)
        storage = deserialized_objects[root_key]
        if view_metadata is not None:
            view_key, offset, view_size = view_metadata
            if view_key not in deserialized_objects:
                deserialized_objects[view_key] = storage[offset:offset +
                                                         view_size]
            return deserialized_objects[view_key]
        else:
            return storage
    else:
        raise RuntimeError("Unknown saved id type: %s" % saved_id[0])


deserialized_objects: Dict[int, Any] = {}


def load(model_path):
    with open(model_path, 'rb') as f:
        f.seek(0)
        magic_number = pickle.load(f, **pickle_load_args)
        if magic_number != MAGIC_NUMBER:
            raise RuntimeError("Invalid magic number; corrupt file?")
        protocol_version = pickle.load(f, **pickle_load_args)
        if protocol_version != PROTOCOL_VERSION:
            raise RuntimeError("Invalid protocol version: %s" %
                               protocol_version)

        _sys_info = pickle.load(f, **pickle_load_args)
        unpickler = pickle.Unpickler(f, **pickle_load_args)
        unpickler.persistent_load = persistent_load
        result = unpickler.load()

        deserialized_storage_keys = pickle.load(f, **pickle_load_args)

        offset = f.tell()
        for key in deserialized_storage_keys:
            assert key in deserialized_objects
            deserialized_objects[key]._set_from_file(f, offset, True)
            if offset is not None:
                offset = f.tell()
    return result


def static_params_to_dygraph(model, static_tensor_dict):
    """Simple tool for convert static paramters to dygraph paramters dict.

    **NOTE** The model must both support static graph and dygraph mode.

    Args:
        model (nn.Layer): the model of a neural network.
        static_tensor_dict (string): path of which locate the saved paramters in static mode.
            Usualy load by `paddle.static.load_program_state`.

    Returns:
        [tensor dict]: a state dict the same as the dygraph mode.
    """
    state_dict = model.state_dict()
    # static_tensor_dict = paddle.static.load_program_state(static_params_path)

    ret_dict = dict()
    for n, p in state_dict.items():
        if p.name not in static_tensor_dict:
            logger.info("%s paramter is missing from you state dict." % n)
            continue
        ret_dict[n] = static_tensor_dict[p.name]

    return ret_dict


def dygraph_params_to_static(model, dygraph_tensor_dict, topo=None):
    """Simple tool for convert dygraph paramters to static paramters dict.

    **NOTE** The model must both support static graph and dygraph mode.

    Args:
        model (nn.Layer): the model of a neural network.
        dygraph_tensor_dict (string): path of which locate the saved paramters in static mode.

    Returns:
        [tensor dict]: a state dict the same as the dygraph mode.
    """
    state_dict = model.state_dict()

    ret_dict = dict()
    for name, parm in state_dict.items():
        if name not in dygraph_tensor_dict:
            logger.info("%s paramter is missing from you state dict." % name)
            continue

        tensor = dygraph_tensor_dict[name]
        if parm.is_distributed:
            assert topo is not None
            for dim, v in enumerate(tensor.shape):
                if parm.shape[dim] != v:
                    break

            splited = np.split(
                tensor, topo.mp_info.size, axis=dim)[topo.mp_info.rank]
            ret_dict[parm.name] = splited
        else:
            ret_dict[parm.name] = tensor

    return ret_dict


class TimeCostAverage(object):
    """
    Simple tool for calcluating time average cost in the process of training and inferencing.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the recoder state, and reset the `cnt` to zero.
        """
        self.cnt = 0
        self.total_time = 0

    def record(self, usetime):
        """
        Recoding the time cost in current step and accumulating the `cnt`.
        """
        self.cnt += 1
        self.total_time += usetime

    def get_average(self):
        """
        Returning the average time cost after the start of training.
        """
        if self.cnt == 0:
            return 0
        return self.total_time / self.cnt


def get_env_device():
    """
    Return the device name of running enviroment.
    """
    if paddle.is_compiled_with_cuda():
        return 'gpu'
    elif paddle.is_compiled_with_npu():
        return 'npu'
    elif paddle.is_compiled_with_rocm():
        return 'rocm'
    elif paddle.is_compiled_with_xpu():
        return 'xpu'
    return 'cpu'


def compare_version(version, pair_version):
    """
    Args:
        version (str): The first version string needed to be compared.
            The format of version string should be as follow : "xxx.yyy.zzz".
        pair_version (str): The second version string needed to be compared.
             The format of version string should be as follow : "xxx.yyy.zzz".
    Returns:
        int: The result of comparasion. 1 means version > pair_version; 0 means
            version = pair_version; -1 means version < pair_version.
    
    Examples:
        >>> compare_version("2.2.1", "2.2.0")
        >>> 1
        >>> compare_version("2.2.0", "2.2.0")
        >>> 0
        >>> compare_version("2.2.0-rc0", "2.2.0")
        >>> -1
        >>> compare_version("2.3.0-rc0", "2.2.0")
        >>> 1
    """
    version = version.strip()
    pair_version = pair_version.strip()
    if version == pair_version:
        return 0
    version_list = version.split(".")
    pair_version_list = pair_version.split(".")
    for version_code, pair_version_code in zip(version_list, pair_version_list):
        if not version_code.isnumeric():
            return -1
        if not pair_version_code.isnumeric():
            return 1
        if int(version_code) > int(pair_version_code):
            return 1
        elif int(version_code) < int(pair_version_code):
            return -1
    return 0
