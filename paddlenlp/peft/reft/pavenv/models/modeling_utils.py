# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import nn

from .constants import CONST_INPUT_HOOK, CONST_OUTPUT_HOOK
from .intervenable_modelcard import type_to_module_mapping


def get_internal_model_type(model):
    """Return the model type."""
    return type(model)


def getattr_for_paddle_module(model, parameter_name):
    """Recursively fetch the model based on the name."""
    current_module = model
    for param in parameter_name.split("."):
        if "[" in param:
            current_module = getattr(current_module, param.split("[")[0])[int(param.split("[")[-1].strip("]"))]
        else:
            current_module = getattr(current_module, param)
    return current_module


def get_module_hook(model, representation) -> nn.Layer:
    """Render the intervening module with a hook."""
    if (
        get_internal_model_type(model) in type_to_module_mapping
        and representation.component in type_to_module_mapping[get_internal_model_type(model)]
    ):
        type_info = type_to_module_mapping[get_internal_model_type(model)][representation.component]
        parameter_name = type_info[0]
        hook_type = type_info[1]

        if "%s" in parameter_name and representation.moe_key is None:
            # we assume it is for the layer.
            parameter_name = parameter_name % (representation.layer)
        elif "%s" in parameter_name and representation.moe_key is not None:
            parameter_name = parameter_name % (
                int(representation.layer),
                int(representation.moe_key),
            )
    else:
        parameter_name = ".".join(representation.component.split(".")[:-1])
        if representation.component.split(".")[-1] == "input":
            hook_type = CONST_INPUT_HOOK
        elif representation.component.split(".")[-1] == "output":
            hook_type = CONST_OUTPUT_HOOK

    module = getattr_for_paddle_module(model, parameter_name)
    module_hook = getattr(module, hook_type)
    return module_hook


class HandlerList:
    """General class to set hooks and set off hooks."""

    def __init__(self, handlers):
        self.handlers = handlers

    def __len__(self):
        return len(self.handlers)

    def remove(self):
        for handler in self.handlers:
            handler.remove()

    def extend(self, new_handlers):
        self.handlers.extend(new_handlers.handlers)
        return self


#  gather hidden states on intervention locations
def gather_neurons(tensor_input, unit_locations_as_list):
    unit_locations = paddle.to_tensor(unit_locations_as_list, place=tensor_input.place)
    tensor_output = paddle.take_along_axis(
        tensor_input,
        axis=1,
        indices=unit_locations.reshape([*unit_locations.shape, *(1,) * (len(tensor_input.shape) - 2)]).expand(
            [-1, -1, *tensor_input.shape[2:]]
        ),
    )
    return tensor_output


# Replace selected neurons in `tensor_input` by `replacing_tensor_input`.
def scatter_neurons(
    tensor_input,
    replacing_tensor_input,
    unit,
    unit_locations_as_list,
):
    unit_locations = paddle.to_tensor(
        unit_locations_as_list,
        place=tensor_input.place,
    )

    # [1,1,4096]
    meta_component = paddle.arange(tensor_input.shape[-1]).unsqueeze(axis=0).unsqueeze(axis=0)

    start_index, end_index = (
        meta_component.min().tolist(),
        meta_component.max().tolist() + 1,
    )
    # 4096
    # last_dim = meta_component.shape[-1]
    # 0, 1, 2, ..., batch_size-1
    _batch_idx = paddle.arange(tensor_input.shape[0]).unsqueeze(1)

    if unit in {"pos"}:
        tensor_input[_batch_idx, unit_locations, start_index:end_index] = replacing_tensor_input
        return tensor_input
    else:
        raise NotImplementedError(f"Unit {unit} not implemented.")


# do intervention
def do_intervention(
    base_representation,
    intervention,
):
    """Do the actual intervention."""
    # base_representation： 从隐藏状态抽取出的对应token的隐藏状态 f7+l7: batch_size, 14, hidden_size
    # intervention: 干预的模型
    # flatten
    original_base_shape = base_representation.shape
    if len(original_base_shape) == 2 or intervention.keep_last_dim:
        base_representation_f = base_representation
    intervened_representation = intervention(
        base_representation_f,
    )
    return intervened_representation
