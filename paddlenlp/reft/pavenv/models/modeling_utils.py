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

import types

import paddle
from paddle import nn

from .constants import CONST_INPUT_HOOK, CONST_OUTPUT_HOOK, split_three
from .intervenable_modelcard import type_to_dimension_mapping, type_to_module_mapping
from .interventions import LocalistRepresentationIntervention


def get_internal_model_type(model):
    """Return the model type."""
    return type(model)


def is_stateless(model):
    """Determine if the model is stateful (e.g., rnn) or stateless (e.g.,
    transformer)
    """
    if is_gru(model):
        return False
    return True


def is_gru(model):
    """Determine if this is a transformer model."""
    # if (
    #     type(model) == GRUModel
    #     or type(model) == GRULMHeadModel
    #     or type(model) == GRUForClassification
    # ):
    #     return True
    return False


def is_mlp(model):
    """Determine if this is a mlp model."""
    # if type(model) == MLPModel or type(model) == MLPForClassification:
    #     return True
    return False


def is_transformer(model):
    """Determine if this is a transformer model."""
    if not is_gru(model) and not is_mlp(model):
        return True
    return False


def print_forward_hooks(main_module):
    """Function to print forward hooks of a module and its sub-modules."""
    for name, submodule in main_module.named_modules():
        if hasattr(submodule, "_forward_hooks") and submodule._forward_hooks:
            print(f"Module: {name if name else 'Main Module'}")
            for hook_id, hook in submodule._forward_hooks.items():
                print(f"  ID: {hook_id}, Hook: {hook}")

        if hasattr(submodule, "_forward_pre_hooks") and submodule._forward_hooks:
            print(f"Module: {name if name else 'Main Module'}")
            for hook_id, hook in submodule._forward_pre_hooks.items():
                print(f"  ID: {hook_id}, Hook: {hook}")


def remove_forward_hooks(main_module: nn.Layer):
    """Function to remove all forward and pre-forward hooks from a layer and
    its sub-layers.
    """

    # Remove forward hooks
    for _, sublayer in main_module.named_sublayers():
        if hasattr(sublayer, "_forward_hooks"):
            hooks = list(sublayer._forward_hooks.keys())  # Get a list of hook IDs
            for hook_id in hooks:
                sublayer._forward_hooks.pop(hook_id)

        # Remove pre-forward hooks
        if hasattr(sublayer, "_forward_pre_hooks"):
            pre_hooks = list(sublayer._forward_pre_hooks.keys())  # Get a list of pre-hook IDs
            for pre_hook_id in pre_hooks:
                sublayer._forward_pre_hooks.pop(pre_hook_id)


def getattr_for_torch_module(model, parameter_name):
    """Recursively fetch the model based on the name."""
    current_module = model
    print("current_module", current_module)
    print(parameter_name.split("."))
    for param in parameter_name.split("."):
        print("param", param)
        if "[" in param:
            current_module = getattr(current_module, param.split("[")[0])[int(param.split("[")[-1].strip("]"))]
        else:
            current_module = getattr(current_module, param)
    return current_module


def get_dimension_by_component(model_type, model_config, component) -> int:
    """Based on the representation, get the aligning dimension size."""

    if component not in type_to_dimension_mapping[model_type]:
        return None

    dimension_proposals = type_to_dimension_mapping[model_type][component]
    for proposal in dimension_proposals:
        if proposal.isnumeric():
            dimension = int(proposal)
        elif "*" in proposal:
            # often constant multiplier with MLP
            dimension = getattr_for_torch_module(model_config, proposal.split("*")[0]) * int(proposal.split("*")[1])
        elif "/" in proposal:
            # often split by head number
            if proposal.split("/")[0].isnumeric():
                numr = int(proposal.split("/")[0])
            else:
                numr = getattr_for_torch_module(model_config, proposal.split("/")[0])

            if proposal.split("/")[1].isnumeric():
                denr = int(proposal.split("/")[1])
            else:
                denr = getattr_for_torch_module(model_config, proposal.split("/")[1])
            dimension = int(numr / denr)
        else:
            # dimension = getattr_for_torch_module(model_config, proposal)
            dimension = model_config[proposal]
        if dimension is not None:
            return dimension

    assert False


def get_module_hook(model, representation) -> nn.Layer:
    """Render the intervening module with a hook."""
    if (
        get_internal_model_type(model) in type_to_module_mapping
        and representation.component in type_to_module_mapping[get_internal_model_type(model)]
    ):
        # print("representation.component", representation.component)
        type_info = type_to_module_mapping[get_internal_model_type(model)][representation.component]
        print("type_info", type_info)
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
        print("parameter name 1：", parameter_name)
    else:
        parameter_name = ".".join(representation.component.split(".")[:-1])
        if representation.component.split(".")[-1] == "input":
            hook_type = CONST_INPUT_HOOK
        elif representation.component.split(".")[-1] == "output":
            hook_type = CONST_OUTPUT_HOOK
        print("parameter name 2：", parameter_name)

    module = getattr_for_torch_module(model, parameter_name)
    print("module:", module)
    module_hook = getattr(module, hook_type)
    print("module_hook is :", module_hook)
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


def bsd_to_b_sd(tensor):
    """Convert a tensor of shape (b, s, d) to (b, s*d)."""
    if tensor is None:
        return tensor
    b, s, d = tensor.shape
    return tensor.reshape([b, s * d])


def b_sd_to_bsd(tensor, s):
    """Convert a tensor of shape (b, s*d) back to (b, s, d)."""
    if tensor is None:
        return tensor
    b, sd = tensor.shape
    d = sd // s
    return tensor.reshape(b, s, d)


def bhsd_to_bs_hd(tensor):
    """Convert a tensor of shape (b, h, s, d) to (b, s, h*d)."""
    if tensor is None:
        return tensor
    b, h, s, d = tensor.shape
    return tensor.permute(0, 2, 1, 3).reshape(b, s, h * d)


def bs_hd_to_bhsd(tensor, h):
    """Convert a tensor of shape (b, s, h*d) back to (b, h, s, d)."""
    if tensor is None:
        return tensor
    b, s, hd = tensor.shape

    d = hd // h

    return tensor.reshape(b, s, h, d).permute(0, 2, 1, 3)


def output_to_subcomponent(output, component, model_type, model_config):
    """Split the raw output to subcomponents if specified in the config.

    :param output: the original output from the model component.
    :param component: types of model component, such as
    "block_output" and "query_output" or it can be direct referece, such as
    "h[0].mlp.act" which we will not splice into any subcomponent.
    :param model_type: Hugging Face Model Type
    :param model_config: Hugging Face Model Config
    """
    subcomponent = output
    if model_type in type_to_module_mapping and component in type_to_module_mapping[model_type]:
        split_last_dim_by = type_to_module_mapping[model_type][component][2:]
        if len(split_last_dim_by) != 0 and len(split_last_dim_by) > 2:
            raise ValueError(f"Unsupported {split_last_dim_by}.")
        for i, (split_fn, param) in enumerate(split_last_dim_by):
            if isinstance(param, str):
                param = get_dimension_by_component(model_type, model_config, param)
            subcomponent = split_fn(subcomponent, param)
    return subcomponent


def gather_neurons(tensor_input, unit, unit_locations_as_list):
    """Gather intervening neurons.

    :param tensor_input: tensors of shape (batch_size, sequence_length, ...) if
    `unit` is "pos" or "h", tensors of shape (batch_size, num_heads,
    sequence_length, ...) if `unit` is "h.pos"
    :param unit: the intervention units to gather. Units could be "h" - head
    number, "pos" - position in the sequence, or "dim" - a particular dimension in
    the embedding space. If intervening multiple units, they are ordered and
    separated by `.`. Currently only support "pos", "h", and "h.pos" units.
    :param unit_locations_as_list: tuple of lists of lists of positions to gather
    in tensor_input, according to the unit.
    :return the gathered tensor as tensor_output
    """
    if unit in {"t"}:
        return tensor_input

    if "." in unit:
        unit_locations = (
            paddle.to_tensor(unit_locations_as_list[0], device=tensor_input.device),
            paddle.to_tensor(unit_locations_as_list[1], device=tensor_input.device),
        )
        # we assume unit_locations is a tuple
        head_unit_locations = unit_locations[0]
        pos_unit_locations = unit_locations[1]

        head_tensor_output = paddle.gather(
            tensor_input,
            1,
            head_unit_locations.reshape(*head_unit_locations.shape, *(1,) * (len(tensor_input.shape) - 2)).expand(
                -1, -1, *tensor_input.shape[2:]
            ),
        )  # b, h, s, d
        d = head_tensor_output.shape[1]
        pos_tensor_input = bhsd_to_bs_hd(head_tensor_output)
        pos_tensor_output = paddle.gather(
            pos_tensor_input,
            1,
            pos_unit_locations.reshape(*pos_unit_locations.shape, *(1,) * (len(pos_tensor_input.shape) - 2)).expand(
                -1, -1, *pos_tensor_input.shape[2:]
            ),
        )  # b, num_unit (pos), num_unit (h)*d
        tensor_output = bs_hd_to_bhsd(pos_tensor_output, d)

        return tensor_output  # b, num_unit (h), num_unit (pos), d
    else:
        # unit_locations = paddle.to_tensor(
        #     unit_locations_as_list, device=tensor_input.device
        # )

        unit_locations = paddle.to_tensor(unit_locations_as_list, place=tensor_input.place)

        # tensor_output = paddle.gather(
        #     tensor_input,
        #     # unit_locations.reshape(
        #     #     *unit_locations.shape, *(1,) * (len(tensor_input.shape) - 2)
        #     # ).expand(-1, -1, *tensor_input.shape[2:]),
        # unit_locations.reshape(
        #     [*unit_locations.shape, *(1,) * (len(tensor_input.shape) - 2)]
        # ).expand([-1, -1, *tensor_input.shape[2:]]),
        #     axis=1,
        # )
        tensor_output = (
            paddle.take_along_axis(
                tensor_input,
                axis=1,
                indices=unit_locations.reshape([*unit_locations.shape, *(1,) * (len(tensor_input.shape) - 2)]).expand(
                    [-1, -1, *tensor_input.shape[2:]]
                ),
            ),
        )

        return tensor_output


def scatter_neurons(
    tensor_input,
    replacing_tensor_input,
    component,
    unit,
    unit_locations_as_list,
    model_type,
    model_config,
    use_fast,
):
    """Replace selected neurons in `tensor_input` by `replacing_tensor_input`.

    :param tensor_input: tensors of shape (batch_size, sequence_length, ...) if
    `unit` is "pos" or "h", tensors of shape (batch_size, num_heads,
    sequence_length, ...) if `unit` is "h.pos"
    :param replacing_tensor_input: tensors of shape (batch_size, sequence_length,
    ...) if `unit` is "pos" or
    "h", tensors of shape (batch_size, num_heads, sequence_length, ...) if
    `unit` is "h.pos".
    :param component: types of intervention representations, such as
    "block_output" and "query_output"
    :param unit: the intervention units to gather. Units could be "h" - head
    number, "pos" - position in the sequence, or "dim" - a particular dimension in
    the embedding space. If intervening multiple units, they are ordered and
    separated by `.`. Currently only support "pos", "h", and "h.pos" units.
    :param unit_locations_as_list: tuple of lists of lists of positions to gather
    in tensor_input, according to the unit.
    :param model_type: Hugging Face Model Type
    :param model_config: Hugging Face Model Config
    :param use_fast: whether to use fast path (TODO: fast path condition)
    :return the in-place modified tensor_input
    """
    if "." in unit:
        # extra dimension for multi-level intervention
        unit_locations = (
            paddle.to_tensor(unit_locations_as_list[0], device=tensor_input.device),
            paddle.to_tensor(unit_locations_as_list[1], device=tensor_input.device),
        )
    else:
        unit_locations = paddle.to_tensor(
            unit_locations_as_list,
            # device=tensor_input.device
            place=tensor_input.place,
        )

    # if tensor is splitted, we need to get the start and end indices
    meta_component = output_to_subcomponent(
        paddle.arange(tensor_input.shape[-1]).unsqueeze(axis=0).unsqueeze(axis=0),
        component,
        model_type,
        model_config,
    )
    start_index, end_index = (
        meta_component.min().tolist(),
        meta_component.max().tolist() + 1,
    )
    # print('start ane end index:', start_index, end_index)
    # 4096
    last_dim = meta_component.shape[-1]
    # 0, 1, 2, ..., batch_size-1
    _batch_idx = paddle.arange(tensor_input.shape[0]).unsqueeze(1)

    # in case it is time step, there is no sequence-related index
    if unit in {"t"}:
        # time series models, e.g., gru
        tensor_input[_batch_idx, start_index:end_index] = replacing_tensor_input
        return tensor_input
    elif unit in {"pos"}:
        if use_fast:
            # maybe this is all redundant, but maybe faster slightly?
            tensor_input[_batch_idx, unit_locations[0], start_index:end_index] = replacing_tensor_input
        else:
            tensor_input[_batch_idx, unit_locations, start_index:end_index] = replacing_tensor_input
        return tensor_input
    elif unit in {"h", "h.pos"}:
        # head-based scattering is only special for transformer-based model
        # replacing_tensor_input: b_s, num_h, s, h_dim -> b_s, s, num_h*h_dim
        old_shape = tensor_input.size()  # b_s, s, -1*num_h*d
        new_shape = tensor_input.size()[:-1] + (
            -1,
            meta_component.shape[1],
            last_dim,
        )  # b_s, s, -1, num_h, d
        # get whether split by QKV
        if (
            component in type_to_module_mapping[model_type]
            and len(type_to_module_mapping[model_type][component]) > 2
            and type_to_module_mapping[model_type][component][2][0] == split_three
        ):
            _slice_idx = type_to_module_mapping[model_type][component][2][1]
        else:
            _slice_idx = 0
        tensor_permute = tensor_input.view(new_shape)  # b_s, s, -1, num_h, d
        tensor_permute = tensor_permute.permute(0, 3, 2, 1, 4)  # b_s, num_h, -1, s, d
        if "." in unit:
            # cannot advance indexing on two columns, thus a single for loop is unavoidable.
            for i in range(unit_locations[0].shape[-1]):
                tensor_permute[
                    _batch_idx, unit_locations[0][:, [i]], _slice_idx, unit_locations[1]
                ] = replacing_tensor_input[:, i]
        else:
            tensor_permute[_batch_idx, unit_locations, _slice_idx] = replacing_tensor_input
        # permute back and reshape
        tensor_output = tensor_permute.permute(0, 3, 2, 1, 4)  # b_s, s, -1, num_h, d
        tensor_output = tensor_output.view(old_shape)  # b_s, s, -1*num_h*d
        return tensor_output
    else:
        if "." in unit:
            # cannot advance indexing on two columns, thus a single for loop is unavoidable.
            for i in range(unit_locations[0].shape[-1]):
                tensor_input[_batch_idx, unit_locations[0][:, [i]], unit_locations[1]] = replacing_tensor_input[:, i]
        else:
            tensor_input[_batch_idx, unit_locations] = replacing_tensor_input
        return tensor_input
    assert False


def do_intervention(base_representation, source_representation, intervention, subspaces):
    """Do the actual intervention."""
    # base_representation： 从隐藏状态抽取出的对应token的隐藏状态 f7+l7: batch_size, 14, hidden_size
    # intervention: LoReintervention()
    # 返回值：
    if isinstance(intervention, types.FunctionType):
        if subspaces is None:
            return intervention(base_representation, source_representation)
        else:
            return intervention(base_representation, source_representation, subspaces)

    num_unit = base_representation.shape[1]

    # flatten
    original_base_shape = base_representation.shape
    if (
        len(original_base_shape) == 2
        or (isinstance(intervention, LocalistRepresentationIntervention))
        or intervention.keep_last_dim
    ):
        # keep_last_dim = True 代码走这里
        # no pos dimension, e.g., gru, or opt-out concate last two dims
        base_representation_f = base_representation
        source_representation_f = source_representation
    elif len(original_base_shape) == 3:
        # b, num_unit (pos), d -> b, num_unit*d
        base_representation_f = bsd_to_b_sd(base_representation)
        source_representation_f = bsd_to_b_sd(source_representation)
    elif len(original_base_shape) == 4:
        # b, num_unit (h), s, d -> b, s, num_unit*d
        base_representation_f = bhsd_to_bs_hd(base_representation)
        source_representation_f = bhsd_to_bs_hd(source_representation)
    else:
        assert False  # what's going on?

    intervened_representation = intervention(base_representation_f, source_representation_f, subspaces)

    # post_d = intervened_representation.shape[-1]

    # unflatten
    if (
        len(original_base_shape) == 2
        or isinstance(intervention, LocalistRepresentationIntervention)
        or intervention.keep_last_dim
    ):
        # no pos dimension, e.g., gru or opt-out concate last two dims
        pass
    elif len(original_base_shape) == 3:
        intervened_representation = b_sd_to_bsd(intervened_representation, num_unit)
    elif len(original_base_shape) == 4:
        intervened_representation = bs_hd_to_bhsd(intervened_representation, num_unit)
    else:
        assert False  # what's going on?

    return intervened_representation


def simple_output_to_subcomponent(output, representation_type, model_config):
    """This is an oversimplied version for demo."""
    return output


def simple_scatter_intervention_output(
    original_output,
    intervened_representation,
    representation_type,
    unit,
    unit_locations,
    model_config,
):
    """This is an oversimplied version for demo."""
    for batch_i, locations in enumerate(unit_locations):
        original_output[batch_i, locations] = intervened_representation[batch_i]


def weighted_average(values, weights):
    if len(values) != len(weights):
        raise ValueError("The length of values and weights must be the same.")

    total = sum(v * w for v, w in zip(values, weights))
    return total / sum(weights)
