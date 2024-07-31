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

import copy
import json
import logging
import os
import types
from collections import OrderedDict
from typing import Dict, Optional

import paddle
from paddle import nn

from .basic_utils import count_parameters, create_directory, get_type_from_string
from .configuration_intervenable_model import IntervenableConfig, RepresentationConfig
from .modeling_utils import (
    HandlerList,
    do_intervention,
    gather_neurons,
    get_internal_model_type,
    get_module_hook,
    scatter_neurons,
)


# Generic intervenable model
class IntervenableModel(nn.Layer):
    def __init__(self, config, model, **kwargs):
        super().__init__()
        if isinstance(config, dict) or isinstance(config, list):
            config = IntervenableConfig(representations=config)
        self.config = config
        intervention_type = config.intervention_types
        self.intervention_types = config.intervention_types
        self.config.model_type = str(type(model))
        self.model_has_grad = False
        # each representation can get a different intervention type
        if type(intervention_type) == list:
            assert len(intervention_type) == len(config.representations)
        self.representations = {}
        self.interventions = {}
        # for the last charactor in intervention name
        self._key_collision_counter = {}
        self._intervention_group = {}
        _original_key_order = []
        # for generate
        self._key_setter_call_counter = {}
        for i, representation in enumerate(config.representations):
            _key = self._get_representation_key(representation)
            if representation.intervention is not None:
                intervention = representation.intervention
            # when load reft model from saved config
            else:
                intervention_function = intervention_type if type(intervention_type) != list else intervention_type[i]
                intervention = intervention_function(**config["intervention_params"])

            module_hook = get_module_hook(model, representation)
            self.representations[_key] = representation
            self.interventions[_key] = (intervention, module_hook)
            _original_key_order += [_key]

            # usually, it's a one time call per
            # hook unless model generates.
            self._key_setter_call_counter[_key] = 0

        self.sorted_keys = _original_key_order
        # assign each key to an unique group based on topological order
        _group_key_inc = 0
        for _key in self.sorted_keys:
            self._intervention_group[_group_key_inc] = [_key]
            _group_key_inc += 1
        # sort group key with ascending order
        self._intervention_group = OrderedDict(sorted(self._intervention_group.items()))
        self.model = model
        self.model_config = model.config
        self.model_type = get_internal_model_type(model)
        self.disable_model_gradients()
        self.trainable_model_parameters = {}

    def _reset_hook_count(self):
        """
        Reset the hook count before any generate call
        """
        self._key_setter_call_counter = dict.fromkeys(self._key_setter_call_counter, 0)

    def __str__(self):
        attr_dict = {
            "model_type": str(self.model_type),
            "intervention_types": str(self.intervention_types),
            "alignabls": self.sorted_keys,
        }
        return json.dumps(attr_dict, indent=4)

    def _get_representation_key(self, representation):
        """
        Provide unique key for each intervention
        """
        l = representation.layer
        c = representation.component
        u = representation.unit
        n = representation.max_number_of_units
        if "." in c:
            # string access for sure
            key_proposal = f"comp.{c}.unit.{u}.nunit.{n}"
        else:
            key_proposal = f"layer.{l}.comp.{c}.unit.{u}.nunit.{n}"
        if key_proposal not in self._key_collision_counter:
            self._key_collision_counter[key_proposal] = 0
        else:
            self._key_collision_counter[key_proposal] += 1
        return f"{key_proposal}#{self._key_collision_counter[key_proposal]}"

    def get_trainable_parameters(self):
        """
        Return trainable params as key value pairs
        """
        ret_params = []
        for k, v in self.interventions.items():
            ret_params += [p for p in v[0].parameters()]
        for p in self.model.parameters():
            if p.requires_grad:
                ret_params += [p]
        return ret_params

    def named_parameters(self, recurse=True, include_sublayers=True):
        """
        The above, but for HuggingFace.
        """
        ret_params = []
        for k, v in self.interventions.items():
            ret_params += [(k + "." + n, p) for n, p in v[0].named_parameters()]
        for n, p in self.model.named_parameters():
            if not p.stop_gradient:
                ret_params += [("model." + n, p)]
        return ret_params

    def enable_model_gradients(self):
        """
        Enable gradient in the model
        """
        # Unfreeze all model weights
        self.model.train()
        for param in self.model.parameters():
            param.stop_gradient = False
        self.model_has_grad = True

    def disable_model_gradients(self):
        """
        Disable gradient in the model
        """
        # Freeze all model weights
        self.model.eval()
        for param in self.model.parameters():
            param.stop_gradient = True
        self.model_has_grad = False

    def save(self, save_directory, intervention_params):
        """
        Save interventions to disk or hub
        """
        create_directory(save_directory)

        saving_config = copy.deepcopy(self.config)
        saving_config.sorted_keys = self.sorted_keys
        saving_config.model_type = str(saving_config.model_type)
        saving_config.intervention_types = []
        saving_config.intervention_dimensions = []
        saving_config.intervention_constant_sources = []

        # handle constant source reprs if passed in.
        serialized_representations = []
        for reprs in saving_config.representations:
            serialized_reprs = {}
            for k, v in reprs._asdict().items():
                if k == "hidden_source_representation":
                    continue
                if k == "source_representation":
                    # hidden flag only set here
                    if v is not None:
                        serialized_reprs["hidden_source_representation"] = True
                    serialized_reprs[k] = None
                elif k == "intervention_type":
                    serialized_reprs[k] = None
                elif k == "intervention":
                    serialized_reprs[k] = None
                else:
                    serialized_reprs[k] = v
            serialized_representations += [RepresentationConfig(**serialized_reprs)]
        saving_config.representations = serialized_representations

        for k, v in self.interventions.items():
            intervention = v[0]
            saving_config.intervention_types += [str(type(intervention))]
            binary_filename = f"intkey_{k}.bin"
            # save intervention binary file
            logging.info(f"Saving trainable intervention to {binary_filename}.")
            paddle.save(
                intervention.state_dict(),
                os.path.join(save_directory, binary_filename),
            )
            if intervention.interchange_dim is None:
                saving_config.intervention_dimensions += [None]
            else:
                saving_config.intervention_dimensions += [intervention.interchange_dim.tolist()]
            saving_config.intervention_constant_sources += [intervention.is_source_constant]

        saving_config["intervention_params"] = intervention_params
        # save metadata config
        saving_config.save_pretrained(save_directory)

    @staticmethod
    def load(
        load_directory,
        model,
    ):
        """
        Load interventions from disk
        """
        # load config
        saving_config = IntervenableConfig.from_pretrained(load_directory)
        casted_intervention_types = []
        for type_str in saving_config.intervention_types:
            casted_intervention_types += [get_type_from_string(type_str)]
        saving_config.intervention_types = casted_intervention_types
        casted_representations = []
        for representation_opts in saving_config.representations:
            casted_representations += [RepresentationConfig(*representation_opts)]
        saving_config.representations = casted_representations
        intervenable = IntervenableModel(saving_config, model)

        # load binary files
        for i, (k, v) in enumerate(intervenable.interventions.items()):
            intervention = v[0]
            binary_filename = f"intkey_{k}.bin"
            intervention.is_source_constant = saving_config.intervention_constant_sources[i]
            intervention.set_interchange_dim(saving_config.intervention_dimensions[i])
            saved_state_dict = paddle.load(os.path.join(load_directory, binary_filename))
            intervention.load_state_dict(saved_state_dict)
        return intervenable

    def save_intervention(self, save_directory, include_model=True):
        """
        Instead of saving the metadata with artifacts, it only saves artifacts such as
        trainable weights. This is not a static method, and returns nothing.
        """
        create_directory(save_directory)

        # save binary files
        for k, v in self.interventions.items():
            intervention = v[0]
            binary_filename = f"intkey_{k}.bin"
            # save intervention binary file
            paddle.save(
                intervention.state_dict(),
                os.path.join(save_directory, binary_filename),
            )

        # save model's trainable parameters as well
        if include_model:
            model_state_dict = {}
            model_binary_filename = "paddle_model.bin"
            for n, p in self.model.named_parameters():
                if not p.stop_gradient:
                    model_state_dict[n] = p
            paddle.save(model_state_dict, os.path.join(save_directory, model_binary_filename))

    def load_intervention(self, load_directory, include_model=True):
        """
        Instead of creating an new object, this function loads existing weights onto
        the current object. This is not a static method, and returns nothing.
        """
        # load binary files
        for i, (k, v) in enumerate(self.interventions.items()):
            intervention = v[0]
            binary_filename = f"intkey_{k}.bin"
            saved_state_dict = paddle.load(os.path.join(load_directory, binary_filename))
            intervention.load_state_dict(saved_state_dict)

        # load model's trainable parameters as well
        if include_model:
            model_binary_filename = "pypaddle_model.bin"
            saved_model_state_dict = paddle.load(os.path.join(load_directory, model_binary_filename))
            self.model.load_state_dict(saved_model_state_dict, strict=False)

    def _gather_intervention_output(self, output, representations_key, unit_locations) -> paddle.Tensor:
        """
        Gather intervening activations from the output based on indices
        """
        if isinstance(output, tuple):
            original_output = output[0].clone()
        else:
            original_output = output.clone()
        if unit_locations is None:
            return original_output

        # gather based on intervention locations
        selected_output = gather_neurons(
            original_output,
            unit_locations,
        )
        return selected_output

    def _scatter_intervention_output(
        self,
        output,
        intervened_representation,
        representations_key,
        unit_locations,
    ) -> paddle.Tensor:
        """
        Scatter in the intervened activations in the output
        """
        # data structure casting
        if isinstance(output, tuple):
            original_output = output[0]
        else:
            original_output = output
        # for non-sequence-based models, we simply replace
        # all the activations.
        if unit_locations is None:
            original_output[:] = intervened_representation[:]
            return original_output

        # component = self.representations[representations_key].component
        unit = self.representations[representations_key].unit

        # scatter in-place
        _ = scatter_neurons(
            original_output,
            intervened_representation,
            unit,
            unit_locations,
        )

        return original_output

    def _intervention_setter(
        self,
        keys,
        unit_locations_base,
    ) -> HandlerList:
        """
        Create a list of setter handlers that will set activations
        """
        handlers = []
        for key_i, key in enumerate(keys):
            intervention, module_hook = self.interventions[key]

            def hook_callback(
                model,
                inputs,
                outputs,
            ):
                is_prompt = self._key_setter_call_counter[key] == 0
                if is_prompt:
                    self._key_setter_call_counter[key] += 1
                if not is_prompt:
                    return

                selected_output = self._gather_intervention_output(outputs, key, unit_locations_base[key_i])

                if not isinstance(self.interventions[key][0], types.FunctionType):
                    if intervention.is_source_constant:
                        intervened_representation = do_intervention(
                            selected_output,
                            intervention,
                        )
                if intervened_representation is None:
                    return

                if isinstance(outputs, tuple):
                    _ = self._scatter_intervention_output(
                        outputs[0],
                        intervened_representation,
                        key,
                        unit_locations_base[key_i],
                    )
                else:
                    _ = self._scatter_intervention_output(
                        outputs,
                        intervened_representation,
                        key,
                        unit_locations_base[key_i],
                    )

            handlers.append(
                module_hook(
                    hook_callback,
                )
            )

        return HandlerList(handlers)

    def _wait_for_forward_with_parallel_intervention(
        self,
        unit_locations,
    ):
        # torch.autograd.set_detect_anomaly(True)
        all_set_handlers = HandlerList([])
        unit_locations_base = unit_locations["sources->base"][1]
        for group_id, keys in self._intervention_group.items():
            for key in keys:
                if (
                    isinstance(self.interventions[key][0], types.FunctionType)
                    or self.interventions[key][0].is_source_constant
                ):
                    set_handlers = self._intervention_setter([key], [unit_locations_base[self.sorted_keys.index(key)]])
                    all_set_handlers.extend(set_handlers)
        return all_set_handlers

    def forward(
        self,
        base,
        unit_locations: Optional[Dict] = None,
        labels: Optional[paddle.Tensor] = None,
        output_original_output: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):

        self._reset_hook_count()
        # if no intervention, return base
        if unit_locations is None and len(self.interventions) == 0:
            return self.model(**base), None

        if unit_locations is not None:
            assert "sources->base" in unit_locations or "base" in unit_locations

        base_outputs = None
        if output_original_output:
            base_outputs = self.model(**base, labels=labels)
        try:
            # intervene
            set_handlers_to_remove = self._wait_for_forward_with_parallel_intervention(unit_locations)
            # run intervened forward
            model_kwargs = {}
            if labels is not None:  # for training
                model_kwargs["labels"] = labels
            if "use_cache" in self.model.config.to_dict():  # for transformer models
                model_kwargs["use_cache"] = use_cache
            counterfactual_outputs = self.model(**base, labels=labels)
            set_handlers_to_remove.remove()
        except Exception as e:
            raise e
        self._reset_hook_count()
        return base_outputs, counterfactual_outputs

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def count_parameters(self, include_model=False):
        total_parameters = 0
        for k, v in self.interventions.items():
            total_parameters += count_parameters(v[0])
        if include_model:
            total_parameters += sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total_parameters

    def generate(
        self,
        base,
        unit_locations: Optional[Dict] = None,
        intervene_on_prompt: bool = False,
        output_original_output: Optional[bool] = False,
        **kwargs,
    ):
        self._reset_hook_count()
        self._intervene_on_prompt = intervene_on_prompt
        base_outputs = None
        if output_original_output or True:
            # returning un-intervened output
            base_outputs = self.model.generate(**base, **kwargs)
        set_handlers_to_remove = None
        try:
            # intervene
            set_handlers_to_remove = self._wait_for_forward_with_parallel_intervention(
                unit_locations,
            )
            # run intervened generate
            counterfactual_outputs = self.model.generate(**base, **kwargs)
            set_handlers_to_remove.remove()
        except Exception as e:
            raise e
        self._reset_hook_count()
        return base_outputs, counterfactual_outputs
