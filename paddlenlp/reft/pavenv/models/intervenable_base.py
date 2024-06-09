import json, logging, types
import numpy as np
from collections import OrderedDict
from typing import List, Optional, Tuple, Union, Dict, Any

from .basic_utils import *
from .modeling_utils import *
from .intervention_utils import *
from .interventions import *
from .configuration_intervenable_model import (
    IntervenableConfig,
    RepresentationConfig,
)
from .interventions import (
    TrainableIntervention,
    SkipIntervention,
    CollectIntervention,
    BoundlessRotatedSpaceIntervention,
)

# from torch import optim
import paddle.optimizer as optim
from transformers import get_linear_schedule_with_warmup
from dataclasses import dataclass
from transformers.utils import ModelOutput
from tqdm import tqdm, trange


@dataclass
class IntervenableModelOutput(ModelOutput):
    original_outputs: Optional[Any] = None
    intervened_outputs: Optional[Any] = None
    collected_activations: Optional[Any] = None


class IntervenableModel(nn.Layer):
    """
    Generic intervenable model. Alignments are specified in the config.
    """

    def __init__(self, config, model, **kwargs):
        super().__init__()
        if isinstance(config, dict) or isinstance(config, list):
            config = IntervenableConfig(representations=config)
        self.config = config

        self.mode = config.mode
        intervention_type = config.intervention_types
        # ldn add
        self.intervention_types = config.intervention_types

        # 判断是否有记忆模型,GRU RNN(有记忆), Transformer(无记忆)
        self.is_model_stateless = is_stateless(model)
        self.config.model_type = str(type(model))  # backfill
        self.use_fast = kwargs["use_fast"] if "use_fast" in kwargs else False

        self.model_has_grad = False
        if self.use_fast:
            logging.warn(
                "Detected use_fast=True means the intervention location "
                "will be static within a batch.\n\nIn case multiple "
                "location tags are passed only the first one will "
                "be considered"
            )
        # each representation can get a different intervention type
        if type(intervention_type) == list:
            assert len(intervention_type) == len(config.representations)

        ###
        # We instantiate intervention_layers at locations.
        # Note that the layer name mentioned in the config is
        # abstract. Not the actual module name of the model.
        #
        # This script will automatically convert abstract
        # name into module name if the model type is supported.
        #
        # To support a new model type, you need to provide a
        # mapping between supported abstract type and module name.
        ###
        self.representations = {}
        self.interventions = {}
        self._key_collision_counter = {}
        self.return_collect_activations = False
        # Flags and counters below are for interventions in the model.generate
        # call. We can intervene on the prompt tokens only, on each generated
        # token, or on a combination of both.
        self._is_generation = False
        self._intervene_on_prompt = None
        self._key_getter_call_counter = {}
        self._key_setter_call_counter = {}
        self._intervention_pointers = {}
        self._intervention_reverse_link = {}

        # hooks are stateful internally, meaning that it's aware of how many times
        # it is called during the execution.
        # TODO: this could be merged with call counter above later.
        self._intervention_state = {}

        # We want to associate interventions with a group to do group-wise interventions.
        self._intervention_group = {}
        _any_group_key = False
        _original_key_order = []
        for i, representation in enumerate(config.representations):
            _key = self._get_representation_key(representation)

            if representation.intervention is not None:
                intervention = representation.intervention
                intervention.use_fast = self.use_fast
            else:
                intervention_function = (
                    intervention_type
                    if type(intervention_type) != list
                    else intervention_type[i]
                )
                all_metadata = representation._asdict()
                component_dim = get_dimension_by_component(
                    get_internal_model_type(model),
                    model.config,
                    representation.component,
                )
                if component_dim is not None:
                    component_dim *= int(representation.max_number_of_units)
                all_metadata["embed_dim"] = component_dim
                all_metadata["use_fast"] = self.use_fast
                intervention = intervention_function(**all_metadata)

            if representation.intervention_link_key in self._intervention_pointers:
                self._intervention_reverse_link[_key] = (
                    f"link#{representation.intervention_link_key}"
                )
                intervention = self._intervention_pointers[
                    representation.intervention_link_key
                ]
            elif representation.intervention_link_key is not None:
                self._intervention_pointers[representation.intervention_link_key] = (
                    intervention
                )
                self._intervention_reverse_link[_key] = (
                    f"link#{representation.intervention_link_key}"
                )

            if isinstance(intervention, CollectIntervention):
                self.return_collect_activations = True

            print("representation", representation)
            module_hook = get_module_hook(model, representation)
            self.representations[_key] = representation
            self.interventions[_key] = (intervention, module_hook)
            self._key_getter_call_counter[_key] = (
                0  # we memo how many the hook is called,
            )
            # usually, it's a one time call per
            # hook unless model generates.
            self._key_setter_call_counter[_key] = 0
            self._intervention_state[_key] = InterventionState(_key)
            _original_key_order += [_key]
            if representation.group_key is not None:
                _any_group_key = True
        if self.config.sorted_keys is not None:
            logging.warn(
                "The key is provided in the config. "
                "Assuming this is loaded from a pretrained module."
            )
        if self.config.sorted_keys is not None or "intervenables_sort_fn" not in kwargs:
            self.sorted_keys = _original_key_order
        else:
            # the key order is independent of group, it is used to read out intervention locations.
            self.sorted_keys = kwargs["intervenables_sort_fn"](
                model, self.representations
            )

        """
        We later use _intervention_group to run actual interventions.
        The order the group by group; and there should not be dependency
        between groups.
        """
        if _any_group_key:
            # In case they are grouped, we would expect the execution order is given
            # by the source inputs.
            _validate_group_keys = []
            for _key in self.sorted_keys:
                representation = self.representations[_key]
                assert representation.group_key is not None
                if representation.group_key in self._intervention_group:
                    self._intervention_group[representation.group_key].append(_key)
                else:
                    self._intervention_group[representation.group_key] = [_key]
                _validate_group_keys += [representation.group_key]
            for i in range(len(_validate_group_keys) - 1):
                if _validate_group_keys[i] > _validate_group_keys[i + 1]:
                    logging.info(
                        f"This is not a valid group key order: {_validate_group_keys}"
                    )
                    raise ValueError(
                        "Must be ascending order. "
                        "Interventions would be performed in order within group as well"
                    )
        else:
            # assign each key to an unique group based on topological order
            _group_key_inc = 0
            for _key in self.sorted_keys:
                self._intervention_group[_group_key_inc] = [_key]
                _group_key_inc += 1
        # sort group key with ascending order
        self._intervention_group = OrderedDict(sorted(self._intervention_group.items()))

        # cached swap-in activations
        self.activations = {}
        # cached swapped activations (hot)
        self.hot_activations = {}

        # temp fields should not be accessed outside
        self._batched_setter_activation_select = {}
        """
        Activations in the future list is ALWAYS causally before
        the vanilla activation list. This field becomes crucial
        if we intervene at the same place multiple times.
        """
        self.model = model
        self.model_config = model.config
        self.model_type = get_internal_model_type(model)
        self.disable_model_gradients()
        self.trainable_model_parameters = {}

    def __str__(self):
        """
        Print out basic info about this intervenable instance
        """
        # attr_dict = {
        #     "model_type": self.model_type,
        #     "intervention_types": self.intervention_types,
        #     "alignabls": self.sorted_keys,
        #     "mode": self.mode,
        # }

        attr_dict = {
            "model_type": str(self.model_type),
            "intervention_types": str(self.intervention_types),
            "alignabls": self.sorted_keys,
            "mode": self.mode,
        }
        print("attr_dict", attr_dict)
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

    def _reset_hook_count(self):
        """
        Reset the hook count before any generate call
        """
        self._key_getter_call_counter = dict.fromkeys(self._key_getter_call_counter, 0)
        self._key_setter_call_counter = dict.fromkeys(self._key_setter_call_counter, 0)
        for k, _ in self._intervention_state.items():
            self._intervention_state[k].reset()

    def _remove_forward_hooks(self):
        """
        Clean up all the remaining hooks before any call
        """
        remove_forward_hooks(self.model)

    def _cleanup_states(self, skip_activation_gc=False):
        """
        Clean up all old in memo states of interventions
        """
        self._is_generation = False
        self._remove_forward_hooks()
        self._reset_hook_count()
        if not skip_activation_gc:
            self.activations.clear()
            self.hot_activations.clear()
            self._batched_setter_activation_select.clear()
        else:
            self.activations = {}
            self.hot_activations = {}
            self._batched_setter_activation_select = {}

    def get_trainable_parameters(self):
        """
        Return trainable params as key value pairs
        """
        ret_params = []
        for k, v in self.interventions.items():
            if isinstance(v[0], TrainableIntervention):
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
            if isinstance(v[0], TrainableIntervention):
                ret_params += [(k + "." + n, p) for n, p in v[0].named_parameters()]
        for n, p in self.model.named_parameters():
            if not p.stop_gradient:
                ret_params += [("model." + n, p)]
        return ret_params

    def get_cached_activations(self):
        """
        Return the cached activations with keys
        """
        return self.activations

    def get_cached_hot_activations(self):
        """
        Return the cached hot activations with linked keys
        """
        return self.hot_activations

    def set_temperature(self, temp: paddle.Tensor):
        """
        Set temperature if needed
        """
        for k, v in self.interventions.items():
            if isinstance(v[0], BoundlessRotatedSpaceIntervention) or isinstance(
                v[0], SigmoidMaskIntervention
            ):
                v[0].set_temperature(temp)

    def enable_model_gradients(self):
        """
        Enable gradient in the model
        """
        # Unfreeze all model weights
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
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

    def disable_intervention_gradients(self):
        """
        Disable gradient in the trainable intervention
        """
        # Freeze all intervention weights
        pass

    def set_device(self, device):
        """
        Set device of interventions and the model
        """
        for k, v in self.interventions.items():
            v[0].to(device)
        self.model.to(device)

    def get_device(self):
        """
        Get device of interventions and the model
        """
        return self.model.device

    def count_parameters(self, include_model=False):
        """
        Set device of interventions and the model
        """
        _linked_key_set = set([])
        total_parameters = 0
        for k, v in self.interventions.items():
            if isinstance(v[0], TrainableIntervention):
                if k in self._intervention_reverse_link:
                    if not self._intervention_reverse_link[k] in _linked_key_set:
                        _linked_key_set.add(self._intervention_reverse_link[k])
                        total_parameters += count_parameters(v[0])
                else:
                    total_parameters += count_parameters(v[0])
        if include_model:
            total_parameters += sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        return total_parameters

    def set_zero_grad(self):
        """
        Set device of interventions and the model
        """
        for k, v in self.interventions.items():
            if isinstance(v[0], TrainableIntervention):
                v[0].zero_grad()

    def zero_grad(self):
        """
        The above, but for HuggingFace.
        """
        for k, v in self.interventions.items():
            if isinstance(v[0], TrainableIntervention):
                v[0].zero_grad()

    def save(
        self, save_directory, save_to_hf_hub=False, hf_repo_name="my-awesome-model"
    ):
        """
        Save interventions to disk or hub
        """
        if save_to_hf_hub:
            from huggingface_hub import HfApi

            api = HfApi()

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
            if (
                isinstance(intervention, TrainableIntervention)
                or intervention.source_representation is not None
            ):
                # logging.info(f"Saving trainable intervention to {binary_filename}.")
                paddle.save(
                    intervention.state_dict(),
                    os.path.join(save_directory, binary_filename),
                )
                if save_to_hf_hub:
                    # push to huggingface hub
                    try:
                        api.create_repo(hf_repo_name)
                    except:
                        logging.info(
                            f"Uploading: {binary_filename}, but skipping creating the repo since "
                            f"either {hf_repo_name} exists or having authentication error."
                        )
                    api.upload_file(
                        path_or_fileobj=os.path.join(save_directory, binary_filename),
                        path_in_repo=binary_filename,
                        repo_id=hf_repo_name,
                        repo_type="model",
                    )
            if intervention.interchange_dim is None:
                saving_config.intervention_dimensions += [None]
            else:
                saving_config.intervention_dimensions += [
                    intervention.interchange_dim.tolist()
                ]
            saving_config.intervention_constant_sources += [
                intervention.is_source_constant
            ]

        # save metadata config
        saving_config.save_pretrained(save_directory)
        if save_to_hf_hub:
            # push to huggingface hub
            try:
                api.create_repo(hf_repo_name)
            except:
                logging.info(
                    f"Uploading the config, Skipping creating the repo since "
                    f"either {hf_repo_name} exists or having authentication error."
                )
            api.upload_file(
                path_or_fileobj=os.path.join(save_directory, "config.json"),
                path_in_repo="config.json",
                repo_id=hf_repo_name,
                repo_type="model",
            )

    @staticmethod
    def load(load_directory, model, local_directory=None, from_huggingface_hub=False):
        """
        Load interventions from disk or hub
        """
        if not os.path.exists(load_directory) or from_huggingface_hub:
            from_huggingface_hub = True

            from huggingface_hub import snapshot_download

            load_directory = snapshot_download(
                repo_id=load_directory,
                local_dir=local_directory,
            )

        # load config
        saving_config = IntervenableConfig.from_pretrained(load_directory)
        # saving_config =
        # <class 'pareft.interventions.LoreftIntervention'>]
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
            intervention.is_source_constant = (
                saving_config.intervention_constant_sources[i]
            )
            intervention.set_interchange_dim(saving_config.intervention_dimensions[i])
            if (
                saving_config.intervention_constant_sources[i]
                # and not isinstance(intervention, ZeroIntervention)
                and not isinstance(intervention, SourcelessIntervention)
            ):
                # logging.warn(f"Loading trainable intervention from {binary_filename}.")
                saved_state_dict = paddle.load(
                    os.path.join(load_directory, binary_filename)
                )
                try:
                    intervention.register_buffer(
                        "source_representation",
                        saved_state_dict["source_representation"],
                    )
                except:
                    intervention.source_representation = saved_state_dict[
                        "source_representation"
                    ]
            elif isinstance(intervention, TrainableIntervention):
                saved_state_dict = paddle.load(
                    os.path.join(load_directory, binary_filename)
                )
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
            if isinstance(intervention, TrainableIntervention):
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
            paddle.save(
                model_state_dict, os.path.join(save_directory, model_binary_filename)
            )

    def load_intervention(self, load_directory, include_model=True):
        """
        Instead of creating an new object, this function loads existing weights onto
        the current object. This is not a static method, and returns nothing.
        """
        # load binary files
        for i, (k, v) in enumerate(self.interventions.items()):
            intervention = v[0]
            binary_filename = f"intkey_{k}.bin"
            if isinstance(intervention, TrainableIntervention):
                saved_state_dict = paddle.load(
                    os.path.join(load_directory, binary_filename)
                )
                intervention.load_state_dict(saved_state_dict)

        # load model's trainable parameters as well
        if include_model:
            model_binary_filename = "pypaddle_model.bin"
            saved_model_state_dict = paddle.load(
                os.path.join(load_directory, model_binary_filename)
            )
            self.model.load_state_dict(saved_model_state_dict, strict=False)

    def _gather_intervention_output(
        self, output, representations_key, unit_locations
    ) -> paddle.Tensor:
        """
        Gather intervening activations from the output based on indices
        """

        if (
            representations_key in self._intervention_reverse_link
            and self._intervention_reverse_link[representations_key]
            in self.hot_activations
        ):
            # hot gather
            # clone is needed here by acting as a different module
            # to avoid gradient conflict.
            #
            # enable the following line when an error is hit
            # torch.autograd.set_detect_anomaly(True)
            selected_output = self.hot_activations[
                self._intervention_reverse_link[representations_key]
            ]
        else:
            # data structure casting
            if isinstance(output, tuple):
                original_output = output[0].clone()
            else:
                original_output = output.clone()
            # for non-sequence models, there is no concept of
            # unit location anyway.
            if unit_locations is None:
                return original_output
            # gather subcomponent
            original_output = output_to_subcomponent(
                original_output,
                self.representations[representations_key].component,
                self.model_type,
                self.model_config,
            )

            # gather based on intervention locations
            selected_output = gather_neurons(
                original_output,
                self.representations[representations_key].unit,
                unit_locations,
            )

        return selected_output[0]

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

        component = self.representations[representations_key].component
        unit = self.representations[representations_key].unit

        # scatter in-place
        _ = scatter_neurons(
            original_output,
            intervened_representation,
            component,
            unit,
            unit_locations,
            self.model_type,
            self.model_config,
            self.use_fast,
        )

        return original_output

    def _intervention_getter(
        self,
        keys,
        unit_locations,
    ) -> HandlerList:
        """
        Create a list of getter handlers that will fetch activations
        """
        handlers = []
        for key_i, key in enumerate(keys):
            intervention, module_hook = self.interventions[key]

            def hook_callback(model, args, kwargs, output=None):
                if self._is_generation:
                    pass
                    # for getter, there is no restriction.
                    # is_prompt = self._key_getter_call_counter[key] == 0
                    # if not self._intervene_on_prompt or is_prompt:
                    #     self._key_getter_call_counter[key] += 1
                    # if self._intervene_on_prompt ^ is_prompt:
                    #     return  # no-op
                if output is None:
                    if len(args) == 0:  # kwargs based calls
                        # PR: https://github.com/frankaging/align-transformers/issues/11
                        # We cannot assume the dict only contain one element
                        output = kwargs[list(kwargs.keys())[0]]
                    else:
                        output = args

                if isinstance(intervention, SkipIntervention):
                    selected_output = self._gather_intervention_output(
                        args[0],  # this is actually the input to the module
                        key,
                        unit_locations[key_i],
                    )
                else:
                    selected_output = self._gather_intervention_output(
                        output, key, unit_locations[key_i]
                    )

                if self.is_model_stateless:
                    # WARNING: might be worth to check the below assertion at runtime,
                    # but commenting it out for now just to avoid confusion.
                    # assert key not in self.activations
                    self.activations[key] = selected_output
                else:
                    state_select_flag = []
                    for unit_location in unit_locations[key_i]:
                        if (
                            self._intervention_state[key].getter_version()
                            in unit_location
                        ):
                            state_select_flag += [True]
                        else:
                            state_select_flag += [False]
                    # for stateful model (e.g., gru), we save extra activations and metadata to do
                    # stateful interventions.
                    self.activations.setdefault(key, []).append(
                        (selected_output, state_select_flag)
                    )
                # set version for stateful models
                self._intervention_state[key].inc_getter_version()

            handlers.append(module_hook(hook_callback, with_kwargs=True))

        return HandlerList(handlers)

    def _tidy_stateful_activations(
        self,
    ):
        _need_tidify = False
        for _, v in self.activations.items():
            if isinstance(v[0], tuple) and isinstance(v[0][1], list):
                _need_tidify = True
                break
        if _need_tidify:
            for k, v in self.activations.items():
                self._tidify_activations = [[] for _ in range(v[0][0].shape[0])]
                for t in range(len(v)):
                    activations_at_t = v[t][0]  # a batched tensor
                    states_at_t = (
                        paddle.tensor(v[t][1]).bool().to(activations_at_t.device)
                    )  # a batched bools
                    selected_activations = activations_at_t[states_at_t]
                    selected_indices = paddle.nonzero(states_at_t).squeeze()
                    if len(selected_indices.shape) == 0:
                        selected_indices = selected_indices.unsqueeze(0)
                    for index, activation in zip(
                        selected_indices, selected_activations
                    ):
                        self._tidify_activations[index].append(activation)
                self.activations[k] = self._tidify_activations

    def _reconcile_stateful_cached_activations(
        self,
        key,
        intervening_activations,
        intervening_unit_locations,
    ):
        """Based on the key, we consolidate activations based on key's state"""
        if key not in self.activations:
            return None

        cached_activations = self.activations[key]
        if self.is_model_stateless:
            # nothing to reconcile if stateless
            return cached_activations

        state_select_flag = []
        for unit_location in intervening_unit_locations:
            if self._intervention_state[key].setter_version() in unit_location:
                state_select_flag += [True]
            else:
                state_select_flag += [False]
        state_select_flag = (
            paddle.tensor(state_select_flag).bool().to(intervening_activations.device)
        )
        selected_indices = paddle.nonzero(state_select_flag).squeeze()
        if len(selected_indices.shape) == 0:
            selected_indices = selected_indices.unsqueeze(0)

        # fill activations with proposed only source activations
        reconciled_activations = []
        for index, select_version in enumerate(
            self._batched_setter_activation_select[key]
        ):
            if index in selected_indices:
                reconciled_activations += [cached_activations[index][select_version]]
            else:
                # WARNING: put a dummy tensor, super danger here but let's trust the code for now.
                reconciled_activations += [
                    paddle.zeros_like(cached_activations[index][0])
                ]
        # increment pointer for those we are actually intervening
        for index in selected_indices:
            self._batched_setter_activation_select[key][index] += 1
        # for non-intervening ones, we copy again from base
        reconciled_activations = paddle.stack(reconciled_activations, dim=0)  # batched
        # reconciled_activations[~state_select_flag] = intervening_activations[~state_select_flag]

        return reconciled_activations

    def _intervention_setter(
        self,
        keys,
        unit_locations_base,
        subspaces,
    ) -> HandlerList:
        """
        Create a list of setter handlers that will set activations
        """
        self._tidy_stateful_activations()

        handlers = []
        for key_i, key in enumerate(keys):
            # print('*'*100)
            # print("key is : ", key)
            intervention, module_hook = self.interventions[key]
            if unit_locations_base[0] is not None:
                self._batched_setter_activation_select[key] = [
                    0 for _ in range(len(unit_locations_base[0]))
                ]  # batch_size

            def hook_callback(model, args, kwargs, output=None):
                if self._is_generation:
                    is_prompt = self._key_setter_call_counter[key] == 0
                    if not self._intervene_on_prompt or is_prompt:
                        self._key_setter_call_counter[key] += 1
                    if self._intervene_on_prompt ^ is_prompt:
                        return  # no-op
                if output is None:
                    if len(args) == 0:  # kwargs based calls
                        # PR: https://github.com/frankaging/align-transformers/issues/11
                        # We cannot assume the dict only contain one element
                        output = kwargs[list(kwargs.keys())[0]]
                    else:
                        # pytorch
                        # output = args
                        # 底层Layer中调用第二个参数是output
                        # hook_result = forward_post_hook(self, inputs, outputs)
                        output = kwargs

                selected_output = self._gather_intervention_output(
                    output, key, unit_locations_base[key_i]
                )
                # TODO: need to figure out why clone is needed
                if not self.is_model_stateless:
                    selected_output = selected_output.clone()

                if isinstance(intervention, CollectIntervention):
                    intervened_representation = do_intervention(
                        selected_output,
                        None,
                        intervention,
                        subspaces[key_i] if subspaces is not None else None,
                    )
                    # fail if this is not a fresh collect
                    assert key not in self.activations

                    self.activations[key] = intervened_representation
                    # no-op to the output

                else:
                    if not isinstance(self.interventions[key][0], types.FunctionType):
                        if intervention.is_source_constant:
                            intervened_representation = do_intervention(
                                selected_output,
                                None,
                                intervention,
                                subspaces[key_i] if subspaces is not None else None,
                            )
                        else:
                            intervened_representation = do_intervention(
                                selected_output,
                                self._reconcile_stateful_cached_activations(
                                    key,
                                    selected_output,
                                    unit_locations_base[key_i],
                                ),
                                intervention,
                                subspaces[key_i] if subspaces is not None else None,
                            )
                    else:
                        # highly unlikely it's a primitive intervention type
                        intervened_representation = do_intervention(
                            selected_output,
                            self._reconcile_stateful_cached_activations(
                                key,
                                selected_output,
                                unit_locations_base[key_i],
                            ),
                            intervention,
                            subspaces[key_i] if subspaces is not None else None,
                        )
                    if intervened_representation is None:
                        return

                    # setter can produce hot activations for shared subspace interventions if linked
                    if key in self._intervention_reverse_link:
                        self.hot_activations[self._intervention_reverse_link[key]] = (
                            intervened_representation.clone()
                        )

                    if isinstance(output, tuple):
                        _ = self._scatter_intervention_output(
                            output[0],
                            intervened_representation,
                            key,
                            unit_locations_base[key_i],
                        )
                    else:
                        _ = self._scatter_intervention_output(
                            output,
                            intervened_representation,
                            key,
                            unit_locations_base[key_i],
                        )

                    self._intervention_state[key].inc_setter_version()

            # handlers.append(module_hook(hook_callback, with_kwargs=True))
            handlers.append(
                module_hook(
                    hook_callback,
                )
            )

        return HandlerList(handlers)

    def _input_validation(
        self,
        base,
        sources,
        unit_locations,
        activations_sources,
        subspaces,
    ):
        """Fail fast input validation"""
        if self.mode == "parallel" and unit_locations is not None:
            assert "sources->base" in unit_locations or "base" in unit_locations
        elif (
            activations_sources is None
            and unit_locations is not None
            and self.mode == "serial"
        ):
            assert "sources->base" not in unit_locations

        # sources may contain None, but length should match
        if sources is not None and not (len(sources) == 1 and sources[0] == None):
            if len(sources) != len(self._intervention_group):
                raise ValueError(
                    f"Source length {len(sources)} is not "
                    f"equal to intervention length {len(self._intervention_group)}."
                )
        elif activations_sources is not None:
            if len(activations_sources) != len(self._intervention_group):
                raise ValueError(
                    f"Source activations length {len(activations_sources)} is not "
                    f"equal to intervention length {len(self._intervention_group)}."
                )

        # if it is stateful models, the passed in activations need to have states
        if not self.is_model_stateless and activations_sources is not None:
            for _, v in activations_sources.items():
                if (
                    isinstance(v, list)
                    and isinstance(v[0], tuple)
                    and isinstance(v[0][1], list) != True
                ):
                    raise ValueError(
                        f"Stateful models need nested activations. See our documentions."
                    )

    def _output_validation(
        self,
    ):
        """Safe guarding the execution by checking memory states"""
        if self.is_model_stateless:
            for k, v in self._intervention_state.items():
                if v.getter_version() > 1 or v.setter_version() > 1:
                    raise Exception(
                        f"For stateless model, each getter and setter "
                        f"should be called only once: {self._intervention_state}"
                    )

    def _flatten_input_dict_as_batch(self, input_dict):
        # we also accept grouped sources, will batch them and provide partition info.
        if not isinstance(input_dict, dict):
            assert isinstance(input_dict, list)
            flatten_input_dict = {}
            for k, v in input_dict[0].items():
                flatten_input_dict[k] = {}
            for i in range(0, len(input_dict)):
                for k, v in input_dict[i].items():
                    flatten_input_dict[k] += [v]
            for k, v in flatten_input_dict.items():
                # flatten as one single batch
                flatten_input_dict[k] = paddle.concat(v, dim=0)
        else:
            flatten_input_dict = input_dict
        return flatten_input_dict

    def _get_partition_size(self, input_dict):
        if not isinstance(input_dict, dict):
            assert isinstance(input_dict, list)
            return len(input_dict)
        else:
            return 1

    def _wait_for_forward_with_parallel_intervention(
        self,
        sources,
        unit_locations,
        activations_sources: Optional[Dict] = None,
        subspaces: Optional[List] = None,
    ):
        # torch.autograd.set_detect_anomaly(True)
        all_set_handlers = HandlerList([])
        unit_locations_sources = unit_locations["sources->base"][0]
        unit_locations_base = unit_locations["sources->base"][1]

        # for each source, we hook in getters to cache activations
        # at each aligning representations
        if activations_sources is None:
            assert len(sources) == len(self._intervention_group)
            for group_id, keys in self._intervention_group.items():
                if sources[group_id] is None:
                    continue  # smart jump for advance usage only
                group_get_handlers = HandlerList([])
                for key in keys:
                    get_handlers = self._intervention_getter(
                        [key],
                        [unit_locations_sources[self.sorted_keys.index(key)]],
                    )
                    group_get_handlers.extend(get_handlers)
                _ = self.model(**sources[group_id])
                group_get_handlers.remove()
        else:
            # simply patch in the ones passed in
            self.activations = activations_sources
            for _, passed_in_key in enumerate(self.activations):
                assert passed_in_key in self.sorted_keys

        # in parallel mode, we swap cached activations all into
        # base at once
        for group_id, keys in self._intervention_group.items():
            for key in keys:
                # skip in case smart jump
                if (
                    key in self.activations
                    or isinstance(self.interventions[key][0], types.FunctionType)
                    or self.interventions[key][0].is_source_constant
                ):
                    set_handlers = self._intervention_setter(
                        [key],
                        [unit_locations_base[self.sorted_keys.index(key)]],
                        # assume same group targeting the same subspace
                        (
                            [subspaces[self.sorted_keys.index(key)]]
                            if subspaces is not None
                            else None
                        ),
                    )
                    # for setters, we don't remove them.
                    all_set_handlers.extend(set_handlers)
        return all_set_handlers

    def _wait_for_forward_with_serial_intervention(
        self,
        sources,
        unit_locations,
        activations_sources: Optional[Dict] = None,
        subspaces: Optional[List] = None,
    ):
        all_set_handlers = HandlerList([])
        for group_id, keys in self._intervention_group.items():
            if sources[group_id] is None:
                continue  # smart jump for advance usage only
            for key_id, key in enumerate(keys):
                if group_id != len(self._intervention_group) - 1:
                    unit_locations_key = f"source_{group_id}->source_{group_id+1}"
                else:
                    unit_locations_key = f"source_{group_id}->base"
                unit_locations_source = unit_locations[unit_locations_key][0][key_id]
                if unit_locations_source is None:
                    continue  # smart jump for advance usage only

                unit_locations_base = unit_locations[unit_locations_key][1][key_id]
                if activations_sources is None:
                    # get activation from source_i
                    get_handlers = self._intervention_getter(
                        [key],
                        [unit_locations_source],
                    )
                else:
                    self.activations[key] = activations_sources[key]
            # call once per group. each intervention is by its own group by default
            if activations_sources is None:
                # this is when previous setter and THEN the getter get called
                _ = self.model(**sources[group_id])
                get_handlers.remove()
                # remove existing setters after getting the curr intervened reprs
                if len(all_set_handlers) > 0:
                    all_set_handlers.remove()
                    all_set_handlers = HandlerList([])

            for key in keys:
                # skip in case smart jump
                if (
                    key in self.activations
                    or isinstance(self.interventions[key][0], types.FunctionType)
                    or self.interventions[key][0].is_source_constant
                ):
                    # set with intervened activation to source_i+1
                    set_handlers = self._intervention_setter(
                        [key],
                        [unit_locations_base],
                        # assume the order
                        (
                            [subspaces[self.sorted_keys.index(key)]]
                            if subspaces is not None
                            else None
                        ),
                    )
                    # for setters, we don't remove them.
                    all_set_handlers.extend(set_handlers)
        return all_set_handlers

    def _broadcast_unit_locations(self, batch_size, unit_locations):
        if unit_locations is None:
            # this means, we don't filter based on location at all.
            return {
                "sources->base": (
                    [None] * len(self.interventions),
                    [None] * len(self.interventions),
                )
            }

        if self.mode == "parallel":
            _unit_locations = {}
            for k, v in unit_locations.items():
                # special broadcast for base-only interventions
                is_base_only = False
                if k == "base":
                    is_base_only = True
                    k = "sources->base"
                if isinstance(v, int):
                    if is_base_only:
                        _unit_locations[k] = (
                            None,
                            [[[v]] * batch_size] * len(self.interventions),
                        )
                    else:
                        _unit_locations[k] = (
                            [[[v]] * batch_size] * len(self.interventions),
                            [[[v]] * batch_size] * len(self.interventions),
                        )
                    self.use_fast = True
                elif len(v) == 2 and isinstance(v[0], int) and isinstance(v[1], int):
                    _unit_locations[k] = (
                        [[[v[0]]] * batch_size] * len(self.interventions),
                        [[[v[1]]] * batch_size] * len(self.interventions),
                    )
                    self.use_fast = True
                elif len(v) == 2 and v[0] == None and isinstance(v[1], int):
                    _unit_locations[k] = (
                        None,
                        [[[v[1]]] * batch_size] * len(self.interventions),
                    )
                    self.use_fast = True
                elif len(v) == 2 and isinstance(v[0], int) and v[1] == None:
                    _unit_locations[k] = (
                        [[[v[0]]] * batch_size] * len(self.interventions),
                        None,
                    )
                    self.use_fast = True
                elif isinstance(v, list) and get_list_depth(v) == 1:
                    # [0,1,2,3] -> [[[0,1,2,3]]], ...
                    if is_base_only:
                        _unit_locations[k] = (
                            None,
                            [[v] * batch_size] * len(self.interventions),
                        )
                    else:
                        _unit_locations[k] = (
                            [[v] * batch_size] * len(self.interventions),
                            [[v] * batch_size] * len(self.interventions),
                        )
                    self.use_fast = True
                else:
                    if is_base_only:
                        _unit_locations[k] = (None, v)
                    else:
                        _unit_locations[k] = v
        elif self.mode == "serial":
            _unit_locations = {}
            for k, v in unit_locations.items():
                if isinstance(v, int):
                    _unit_locations[k] = (
                        [[[v]] * batch_size] * len(self.interventions),
                        [[[v]] * batch_size] * len(self.interventions),
                    )
                    self.use_fast = True
                elif len(v) == 2 and isinstance(v[0], int) and isinstance(v[1], int):
                    _unit_locations[k] = (
                        [[[v[0]]] * batch_size] * len(self.interventions),
                        [[[v[1]]] * batch_size] * len(self.interventions),
                    )
                    self.use_fast = True
                elif len(v) == 2 and v[0] == None and isinstance(v[1], int):
                    _unit_locations[k] = (
                        None,
                        [[[v[1]]] * batch_size] * len(self.interventions),
                    )
                    self.use_fast = True
                elif len(v) == 2 and isinstance(v[0], int) and v[1] == None:
                    _unit_locations[k] = (
                        [[[v[0]]] * batch_size] * len(self.interventions),
                        None,
                    )
                    self.use_fast = True
                elif isinstance(v, list) and get_list_depth(v) == 1:
                    # [0,1,2,3] -> [[[0,1,2,3]]], ...
                    _unit_locations[k] = (
                        [[v] * batch_size] * len(self.interventions),
                        [[v] * batch_size] * len(self.interventions),
                    )
                    self.use_fast = True
                else:
                    _unit_locations[k] = v
        else:
            raise ValueError(f"The mode {self.mode} is not supported.")
        return _unit_locations

    def _broadcast_source_representations(self, source_representations):
        """Broadcast simple inputs to a dict"""
        _source_representations = {}
        if isinstance(source_representations, dict) or source_representations is None:
            # pass to broadcast for advance usage
            _source_representations = source_representations
        elif isinstance(source_representations, list):
            for i, key in enumerate(self.sorted_keys):
                _source_representations[key] = source_representations[i]
        elif isinstance(source_representations, paddle.Tensor):
            for key in self.sorted_keys:
                _source_representations[key] = source_representations
        else:
            raise ValueError(
                "Accept input type for source_representations is [Dict, List, paddle.Tensor]"
            )
        return _source_representations

    def _broadcast_sources(self, sources):
        """Broadcast simple inputs to a dict"""
        _sources = sources
        if len(sources) == 1 and len(self._intervention_group) > 1:
            for _ in range(len(self._intervention_group) - 1):
                _sources += [sources[0]]
        else:
            _sources = sources
        return _sources

    def _broadcast_subspaces(self, batch_size, subspaces):
        """Broadcast simple subspaces input"""
        _subspaces = subspaces
        if isinstance(subspaces, int):
            _subspaces = [[[subspaces]] * batch_size] * len(self.interventions)

        elif isinstance(subspaces, list) and isinstance(subspaces[0], int):
            _subspaces = [[subspaces] * batch_size] * len(self.interventions)
        else:
            # TODO: subspaces is easier to add more broadcast majic.
            pass
        return _subspaces

    def forward(
        self,
        base,
        sources: Optional[List] = None,
        unit_locations: Optional[Dict] = None,
        source_representations: Optional[Dict] = None,
        subspaces: Optional[List] = None,
        labels: Optional[paddle.Tensor] = None,
        output_original_output: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = True,
    ):
        """
        Main forward function that serves a wrapper to
        actual model forward calls. It will use forward
        hooks to do interventions.

        In essense, sources will lead to getter hooks to
        get activations. We will use these activations to
        intervene on our base example.

        Parameters:
        base:                The base example.
        sources:             A list of source examples.
        unit_locations:      The intervention locations.
        activations_sources: A list of representations.
        subspace:            Subspace interventions.

        Return:
        base_output: the non-intervened output of the base
        input.
        counterfactual_outputs: the intervened output of the
        base input.

        Notes:

        1) unit_locations
        unit_locations is a dict where keys are tied with
        example pairs involved in one intervention as,
        {
            "sources->base" : List[]
        }

        the shape can be

        2 * num_intervention * bs * num_max_unit

        OR

        2 * num_intervention * num_intervention_level * bs * num_max_unit

        if we intervene on h.pos which is a nested intervention location.

        2) subspaces
        subspaces is a list of indices indicating which subspace will
        this intervention target given an example in the batch.

        An intervention could be initialized with subspace parition as,
        [[... subspace_1 ...], [... subspace_2 ...], [rest]]

        An intervention may be targeting a specific partition.

        This input field should look like something like,
        [
            [[subspace indices], [subspace indices]], <- for the first intervention
            None,                                     <- for the second intervention
            [[subspace indices], [subspace indices]]
        ]

        Only setter (where do_intervention is called) needs this field.

        *We assume base and source targetting the same subspace for now.
        *We assume only a single space is targeted for now (although 2d list is provided).

        Since we now support group-based intervention, the number of sources
        should be equal to the total number of groups.
        """
        # TODO: forgive me now, i will change this later.
        activations_sources = source_representations
        if sources is not None and not isinstance(sources, list):
            sources = [sources]

        self._cleanup_states()

        # if no source input or intervention, we return base
        if (
            sources is None
            and activations_sources is None
            and unit_locations is None
            and len(self.interventions) == 0
        ):
            return self.model(**base), None
        # broadcast unit_locations from layer_nums * batch_size * inter_token_nums to layer_nums * batch_size * inter_token_nums
        unit_locations = self._broadcast_unit_locations(
            get_batch_size(base), unit_locations
        )
        sources = [None] * len(self._intervention_group) if sources is None else sources
        sources = self._broadcast_sources(sources)
        activations_sources = self._broadcast_source_representations(
            activations_sources
        )
        subspaces = self._broadcast_subspaces(get_batch_size(base), subspaces)

        self._input_validation(
            base,
            sources,
            unit_locations,
            activations_sources,
            subspaces,
        )

        base_outputs = None
        if output_original_output:
            # returning un-intervened output with gradients
            # base_outputs = self.model(**base)
            # self.model = LlamaForCausalLM()
            base_outputs = self.model(**base, labels=labels)
        # print("base_outputs is ", base_outputs[0])

        # return 0, self.model(**base, labels=labels)

        try:
            # intervene
            if self.mode == "parallel":
                set_handlers_to_remove = (
                    self._wait_for_forward_with_parallel_intervention(
                        sources,
                        unit_locations,
                        activations_sources,
                        subspaces,
                    )
                )
            elif self.mode == "serial":
                set_handlers_to_remove = (
                    self._wait_for_forward_with_serial_intervention(
                        sources,
                        unit_locations,
                        activations_sources,
                        subspaces,
                    )
                )

            # run intervened forward
            model_kwargs = {}
            if labels is not None:  # for training
                model_kwargs["labels"] = labels
            if "use_cache" in self.model.config.to_dict():  # for transformer models
                model_kwargs["use_cache"] = use_cache

            # counterfactual_outputs = self.model(**base, **model_kwargs)
            counterfactual_outputs = self.model(**base, labels=labels)
            # print("counterfactual_outputs", counterfactual_outputs[0])
            # print("=" * 100)
            # exit(0)
            set_handlers_to_remove.remove()

            self._output_validation()

            collected_activations = []
            if self.return_collect_activations:
                for key in self.sorted_keys:
                    if isinstance(self.interventions[key][0], CollectIntervention):
                        collected_activations += self.activations[key]

        except Exception as e:
            raise e
        finally:
            self._cleanup_states(
                skip_activation_gc=(sources is None and activations_sources is not None)
                or self.return_collect_activations
            )

        if self.return_collect_activations:
            if return_dict:
                return IntervenableModelOutput(
                    original_outputs=base_outputs,
                    intervened_outputs=counterfactual_outputs,
                    collected_activations=collected_activations,
                )

            return (base_outputs, collected_activations), counterfactual_outputs

        if return_dict:
            return IntervenableModelOutput(
                original_outputs=base_outputs,
                intervened_outputs=counterfactual_outputs,
                collected_activations=None,
            )

        return base_outputs, counterfactual_outputs

    def generate(
        self,
        base,
        sources: Optional[List] = None,
        unit_locations: Optional[Dict] = None,
        source_representations: Optional[Dict] = None,
        intervene_on_prompt: bool = False,
        subspaces: Optional[List] = None,
        output_original_output: Optional[bool] = False,
        **kwargs,
    ):
        """
        Intervenable generation function that serves a
        wrapper to regular model generate calls.

        Currently, we support basic interventions **in the
        prompt only**. We will support generation interventions
        in the next release.

        TODO: Unroll sources and intervene in the generation step.

        Parameters:
        base:                The base example.
        sources:             A list of source examples.
        unit_locations:      The intervention locations of
                             base.
        activations_sources: A list of representations.
        intervene_on_prompt: Whether only intervene on prompt.
        **kwargs:            All other generation parameters.

        Return:
        base_output: the non-intervened output of the base
        input.
        counterfactual_outputs: the intervened output of the
        base input.
        """
        # TODO: forgive me now, i will change this later.
        activations_sources = source_representations
        if sources is not None and not isinstance(sources, list):
            sources = [sources]

        self._cleanup_states()

        self._intervene_on_prompt = intervene_on_prompt
        self._is_generation = True

        if not intervene_on_prompt and unit_locations is None:
            # that means, we intervene on every generated tokens!
            unit_locations = {"base": 0}

        # broadcast
        unit_locations = self._broadcast_unit_locations(
            get_batch_size(base), unit_locations
        )
        sources = [None] * len(self._intervention_group) if sources is None else sources
        sources = self._broadcast_sources(sources)
        activations_sources = self._broadcast_source_representations(
            activations_sources
        )
        subspaces = self._broadcast_subspaces(get_batch_size(base), subspaces)

        self._input_validation(
            base,
            sources,
            unit_locations,
            activations_sources,
            subspaces,
        )

        base_outputs = None
        if output_original_output:
            # returning un-intervened output
            base_outputs = self.model.generate(**base, **kwargs)
            print("base_outputs is", base_outputs)

        set_handlers_to_remove = None
        try:
            # intervene
            if self.mode == "parallel":
                set_handlers_to_remove = (
                    self._wait_for_forward_with_parallel_intervention(
                        sources,
                        unit_locations,
                        activations_sources,
                        subspaces,
                    )
                )
            elif self.mode == "serial":
                set_handlers_to_remove = (
                    self._wait_for_forward_with_serial_intervention(
                        sources,
                        unit_locations,
                        activations_sources,
                        subspaces,
                    )
                )

            # run intervened generate
            counterfactual_outputs = self.model.generate(**base, **kwargs)

            collected_activations = []
            if self.return_collect_activations:
                for key in self.sorted_keys:
                    if isinstance(self.interventions[key][0], CollectIntervention):
                        collected_activations += self.activations[key]
        except Exception as e:
            raise e
        finally:
            if set_handlers_to_remove is not None:
                set_handlers_to_remove.remove()
            self._is_generation = False
            self._cleanup_states(
                skip_activation_gc=(sources is None and activations_sources is not None)
                or self.return_collect_activations
            )

        if self.return_collect_activations:
            return (base_outputs, collected_activations), counterfactual_outputs

        return base_outputs, counterfactual_outputs

    def _batch_process_unit_location(self, inputs):
        """
        Convert original data batch according
        to the intervenable settings.

        The function respects inputs in the following
        data format.


        Each location list in the raw input as,

        [[i, j, ...], [m, n, ...], ...] batched
        where i, j are the unit index, the outter
        list is for the batch


        Possible fields in the input:

        inputs["source_0->base.0.pos"] -> batched
        inputs["source_0->base.1.pos"] -> batched
        AND
        inputs["source_0->source_1.0.pos"] -> batched
        inputs["source_0->source_1.1.pos"] -> batched
        ...

        multiple source locations are included in case
        there are multiple sources.

        We also need to consider whether we are doing
        parallel or serial interventions.

        We also need to consider the granularity. In case
        we are intervening h.pos, which is a specific location
        in a specific head:

        inputs["source_0->base.0.pos"] -> batched
        inputs["source_0->source_1.0.h"] -> batched

        inputs["source_0->base.0.pos"] -> batched
        inputs["source_0->source_1.0.pos"] -> batched
        """
        batched_location_dict = {}

        _source_ind = []
        for k, _ in inputs.items():
            if "->" in k:
                for sub_k in k.split("->"):
                    if "source" in sub_k:
                        _source_ind += [int(sub_k.split("_")[1])]
        _max_source_ind = max(_source_ind)

        # we assume source_0 -> source_1, ..., source_last -> base
        # each pair uses an intervention

        if self.mode == "parallel":
            # all source into base at once but may engage different locations
            _curr_source_ind = 0
            _parallel_aggr_left = []
            _parallel_aggr_right = []
            for _, rep in self.representations.items():
                _curr_source_ind_inc = _curr_source_ind + 1
                _prefix = f"source_{_curr_source_ind}->base"
                _prefix_left = f"{_prefix}.0"
                _prefix_right = f"{_prefix}.1"
                _sub_loc_aggr_left = []  # 3d
                _sub_loc_aggr_right = []  # 3d
                for sub_loc in rep.unit.split("."):
                    _sub_loc_aggr_left += [inputs[f"{_prefix_left}.{sub_loc}"]]
                    _sub_loc_aggr_right += [inputs[f"{_prefix_right}.{sub_loc}"]]
                if len(rep.unit.split(".")) == 1:
                    _sub_loc_aggr_left = _sub_loc_aggr_left[0]
                    _sub_loc_aggr_right = _sub_loc_aggr_right[0]
                _parallel_aggr_left += [_sub_loc_aggr_left]  # 3D or 4D
                _parallel_aggr_right += [_sub_loc_aggr_right]  # 3D or 4D
                _curr_source_ind += 1

            batched_location_dict["sources->base"] = (
                _parallel_aggr_left,
                _parallel_aggr_right,
            )

        else:
            # source into another source and finally to the base engaging different locations
            _curr_source_ind = 0
            for _, rep in self.representations.items():
                _curr_source_ind_inc = _curr_source_ind + 1
                _prefix = (
                    f"source_{_curr_source_ind}->base"
                    if _curr_source_ind + 1 == len(self.representations)
                    else f"source_{_curr_source_ind}->source{_curr_source_ind_inc}"
                )
                _prefix_left = f"{_prefix}.0"
                _prefix_right = f"{_prefix}.1"
                _sub_loc_aggr_left = []  # 3d
                _sub_loc_aggr_right = []  # 3d
                for sub_loc in rep.unit.split("."):
                    _sub_loc_aggr_left += [inputs[f"{_prefix_left}.{sub_loc}"]]
                    _sub_loc_aggr_right += [inputs[f"{_prefix_right}.{sub_loc}"]]
                if len(rep.unit.split(".")) == 1:
                    _sub_loc_aggr_left = _sub_loc_aggr_left[0]
                    _sub_loc_aggr_right = _sub_loc_aggr_right[0]
                _curr_source_ind += 1
                batched_location_dict[_prefix] = (
                    [_sub_loc_aggr_left],  # 3D or 4D
                    [_sub_loc_aggr_right],  # 3D or 4D
                )

        return batched_location_dict

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def train_alignment(
        self,
        train_dataloader,
        compute_loss,
        compute_metrics,
        inputs_collator,
        **kwargs,
    ):
        """
        The method find alignment.

        a.k.a. training the intervention

        Notes:
        1) we use Adam, and linear lr scheduling.
        2) you can pass in lr or using default 1e-3
        """
        # preprocess basic kwargs
        lr = kwargs["lr"] if "lr" in kwargs else 1e-3
        epochs = kwargs["epochs"] if "epochs" in kwargs else 10
        warm_up_steps = kwargs["warm_up_steps"] if "warm_up_steps" in kwargs else 0.1
        gradient_accumulation_steps = (
            kwargs["gradient_accumulation_steps"]
            if "gradient_accumulation_steps" in kwargs
            else 1
        )

        # some deeper kwargs
        t_total = int(len(train_dataloader) * epochs)
        warm_up_steps = 0.1 * t_total
        target_total_step = len(train_dataloader) * epochs
        optimizer_params = [{"params": self.get_trainable_parameters()}]
        optimizer = (
            kwargs["optimizer"]
            if "optimizer" in kwargs
            else optim.Adam(optimizer_params, lr=lr)
        )
        scheduler = (
            kwargs["scheduler"]
            if "scheduler" in kwargs
            else get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total
            )
        )

        # in case we need additional temp scheduling
        temperature_start = 50.0
        temperature_end = 0.1
        temperature_schedule = (
            paddle.linspace(temperature_start, temperature_end, target_total_step)
            .to(paddle.bfloat16)
            .to(self.get_device())
        )

        # train main loop
        remove_forward_hooks(self.model)
        self.model.eval()  # train enables drop-off but no grads
        epoch_iterator = trange(0, int(epochs), desc="Epoch")
        total_step = 0
        for epoch in epoch_iterator:
            for step, inputs in enumerate(train_dataloader):
                if inputs_collator is not None:
                    inputs = inputs_collator(inputs)
                b_s = inputs["input_ids"].shape[0]
                unit_location_dict = self._batch_process_unit_location(inputs)
                _, counterfactual_outputs = self(
                    {"input_ids": inputs["input_ids"]},
                    [{"input_ids": inputs["source_input_ids"]}],
                    unit_location_dict,
                    subspaces=inputs["subspaces"] if "subspaces" in inputs else None,
                )
                eval_metrics = compute_metrics(
                    [counterfactual_outputs.logits], [inputs["labels"]]
                )

                # loss and backprop
                loss = compute_loss(counterfactual_outputs.logits, inputs["labels"])
                loss_str = round(loss.item(), 2)
                epoch_iterator.set_postfix({"loss": loss_str, "acc": eval_metrics})

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                loss.backward()
                if total_step % gradient_accumulation_steps == 0:
                    if not (gradient_accumulation_steps > 1 and total_step == 0):
                        optimizer.step()
                        scheduler.step()
                        self.set_zero_grad()
                        self.set_temperature(temperature_schedule[total_step])
                total_step += 1

    def eval_alignment(
        self,
        eval_dataloader,
        compute_metrics,
        inputs_collator,
        **kwargs,
    ):
        """
        The method evaluate alignment.
        """

        all_metrics = []
        all_num_examples = []
        paddle.cuda.empty_cache()
        with paddle.no_grad():
            for inputs in tqdm(eval_dataloader, desc="Evaluating", leave=False):
                if inputs_collator is not None:
                    inputs = inputs_collator(inputs)
                b_s = inputs["input_ids"].shape[0]
                unit_location_dict = self._batch_process_unit_location(
                    inputs,
                )
                _, counterfactual_outputs = self(
                    {"input_ids": inputs["input_ids"]},
                    [{"input_ids": inputs["source_input_ids"]}],
                    unit_location_dict,
                    subspaces=inputs["subspaces"] if "subspaces" in inputs else None,
                )
                eval_metrics = compute_metrics(
                    [counterfactual_outputs.logits], [inputs["labels"]]
                )
                all_metrics += [eval_metrics]
                all_num_examples += [b_s]
        result = weighted_average(all_metrics, all_num_examples)

        return result


class DistributedRepresentationIntervention(nn.Layer):
    """Distributed representation."""

    def __init__(self, **kwargs):
        super().__init__()
        self.is_repr_distributed = True


class BoundlessRotatedSpaceIntervention(
    TrainableIntervention, DistributedRepresentationIntervention
):
    """Intervention in the rotated space with boundary mask."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rotate_layer = RotateLayer(self.embed_dim)
        self.rotate_layer = paddle.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.intervention_boundaries = paddle.nn.Parameter(
            paddle.tensor([0.5]), requires_grad=True
        )
        self.temperature = paddle.nn.Parameter(paddle.tensor(50.0))
        self.intervention_population = paddle.nn.Parameter(
            paddle.arange(0, self.embed_dim), requires_grad=False
        )

    def get_boundary_parameters(self):
        return self.intervention_boundaries

    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temp: paddle.Tensor):
        self.temperature.data = temp

    def set_intervention_boundaries(self, intervention_boundaries):
        self.intervention_boundaries = paddle.nn.Parameter(
            paddle.tensor([intervention_boundaries]), requires_grad=True
        )

    def forward(self, base, source, subspaces=None):
        batch_size = base.shape[0]
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        # get boundary
        intervention_boundaries = paddle.clamp(self.intervention_boundaries, 1e-3, 1)
        boundary_mask = sigmoid_boundary(
            self.intervention_population.repeat(batch_size, 1),
            0.0,
            intervention_boundaries[0] * int(self.embed_dim),
            self.temperature,
        )
        boundary_mask = (
            paddle.ones(batch_size, device=base.device).unsqueeze(dim=-1) * boundary_mask
        )
        boundary_mask = boundary_mask.to(rotated_base.dtype)
        # interchange
        rotated_output = (
            1.0 - boundary_mask
        ) * rotated_base + boundary_mask * rotated_source
        # inverse output
        output = paddle.matmul(rotated_output, self.rotate_layer.weight.T)
        return output.to(base.dtype)

    def __str__(self):
        return f"BoundlessRotatedSpaceIntervention()"
