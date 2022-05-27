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

import paddle.fluid.core as core


def to_strings_tensor(string_values, name):
    """
    Create the tensor that the value holds the list of string.
    NOTICE: The value will be holded in the cpu place. 
 
    Args:
        string_values(list[string]): The value will be setted to the tensor.
        name(string): The name of the tensor.
    """
    tensor = paddle.Tensor(core.VarDesc.VarType.STRINGS, [], name,
                           core.VarDesc.VarType.STRINGS, False)
    tensor.value().set_string_list(string_values)
    return tensor


def to_map_tensor(string_dict, name):
    """
    Create the tensor that the value holds the map, the type of key is the string.
    NOTICE: The value will be holded in the cpu place. 
 
    Args:
        string_dict(dict): The value will be setted to the tensor.
        name(string): The name of the tensor.
    """
    tensor = paddle.Tensor(core.VarDesc.VarType.MAP, [], name,
                           core.VarDesc.VarType.MAP, True)
    tensor.value().set_string_map(string_dict)
    return tensor
