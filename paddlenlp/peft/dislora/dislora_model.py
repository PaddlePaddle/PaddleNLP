# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import os
import re
from collections import OrderedDict
from typing import Dict, Union

import numpy as np
import paddle
import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import PipelineLayer

from paddlenlp.transformers import AutoConfig, PretrainedModel
from paddlenlp.transformers.model_utils import _add_variant, dtype_guard
from paddlenlp.utils.log import logger

from ...utils.env import DISLORA_WEIGHTS_NAME
from .dislora_config import DisLoRAConfig


def get_dislora_layers():
    from .dislora_layer import DisLoRALinear

    return {
        "DisLoRALinear": DisLoRALinear,
    }


dislora_layers = get_dislora_layers()
DisLoRALinear = dislora_layers["DisLoRALinear"]
AVAILABLE_LAYERS = [
    DisLoRALinear,
]


class DisLoRAModel(nn.Layer):
    restore_layer_map: Dict[nn.Layer, nn.Layer] = {
        DisLoRALinear: nn.Linear,
    }

    def __init__(self, model, dislora_config: DisLoRAConfig) -> None:
        super().__init__()
        self.model_config = AutoConfig.from_pretrained(dislora_config.base_model_name_or_path)
        self.quantized = False
        self.dislora_config = dislora_config
        self.dislora_split_mapping = {}
        if self.dislora_config.dtype is None:
            self.dislora_config.dtype = paddle.get_default_dtype()
        with dtype_guard(self.dislora_config.dtype):
            self.model = self.get_dislora_model(model, dislora_config)
        self.is_pipelinemodel = False
        if issubclass(type(self.model), PipelineLayer):
            raise NotImplementedError("dislora don't support pipeline parallel now")
        if dislora_config.tensor_parallel_degree > 1:
            self.dislora_config.tensor_parallel_degree = -1
            self.model.config.tensor_parallel_degree = -1
            raise NotImplementedError("dislora don't support tensor parallel now")
        # currently tensor_parallel_degree should all be set to -1.
        self.forward = self.model.forward

        logger.info("Mark only dislora and trainable_module as trainable.")
        self.mark_only_dislora_as_trainable()

    @classmethod
    def from_pretrained(cls, model, dislora_path, **kwargs):
        dislora_config = kwargs.pop("dislora_config", None)
        # init dislora config & dislora model
        if not isinstance(dislora_config, DisLoRAConfig):
            dislora_config = DisLoRAConfig.from_pretrained(dislora_path)
        # define a new variable to conserve original lora_config.tensor_parallel_degree value which will update while initializing lora model
        dislora_config_tensor_parallel_degree = dislora_config.tensor_parallel_degree
        dislora_model = cls(model, dislora_config)

        # define dislora weight name
        dislora_weight_name = DISLORA_WEIGHTS_NAME

        # load and set dislora weight parameter
        dislora_weight_path = os.path.join(dislora_path, dislora_weight_name)
        if os.path.exists(dislora_weight_path):
            # load dislora weight parameter
            dislora_state_dict = paddle.load(dislora_weight_path, return_numpy=True)
            logger.info(f"Loading the DisLoRA weights from {dislora_weight_path}")

            if (
                dislora_config_tensor_parallel_degree > 1
                and dislora_config_tensor_parallel_degree != model.config.tensor_parallel_degree
            ):
                raise NotImplementedError(
                    f"{dislora_config_tensor_parallel_degree} is not equal to {model.config.tensor_parallel_degree}. Please merge DisLoRA weights first."
                )
            # set dislora state dict
            dislora_model.set_state_dict(dislora_state_dict)
        else:
            logger.error(f"DisLoRA weights not found under {dislora_path}, creating DisLoRA weights from scratch")

        return dislora_model

    def set_state_dict(self, state_dict):
        import warnings

        warnings.filterwarnings(
            action="ignore", message=".*Skip loading for.*", category=Warning, lineno=0, append=False
        )
        self.model.set_state_dict(state_dict)
        logger.info("Load dislora weight successfully")

    def save_pretrained(self, save_directory: str, merge_tensor_parallel: bool = False, **kwargs):
        logger.info("save dislora pretrained")
        save_model_config = kwargs.get("save_model_config", True)

        variant = kwargs.get("variant", None)
        is_main_process = kwargs.get("is_main_process", paddle.distributed.get_rank() == 0)

        assert not os.path.isfile(
            save_directory
        ), f"Saving directory ({save_directory}) should be a directory, not a file"
        os.makedirs(save_directory, exist_ok=True)

        dislora_config_to_save = DisLoRAConfig(**self.dislora_config.to_dict())
        trainable_state_dict = self.get_trainable_state_dict()

        # save dislora weight
        dislora_weight_name = _add_variant(DISLORA_WEIGHTS_NAME, variant)
        weight_filename = os.path.join(save_directory, dislora_weight_name)
        paddle.save(trainable_state_dict, weight_filename)

        # save dislora config
        if is_main_process:
            dislora_config_to_save.save_pretrained(save_directory)
            if save_model_config:
                model_config_to_save = copy.deepcopy(self.model.config)
                if merge_tensor_parallel:
                    model_config_to_save.tensor_parallel_degree = -1
                model_config_to_save.save_pretrained(save_directory)

    def _find_and_replace_module(self, model, module_name, dislora_config):

        if any(dislora_keyword in module_name.lower() for dislora_keyword in ["dislora", "direc_"]):
            logger.debug(f"Skipping {module_name} - appears to be a DisLoRA submodule")
            return

        try:
            parent_module = model
            attribute_chain = module_name.split(".")
            for name in attribute_chain[:-1]:
                parent_module = getattr(parent_module, name)
            module = getattr(parent_module, attribute_chain[-1])
        except AttributeError as e:
            logger.error(f"Cannot access module {module_name}: {e}")
            raise ValueError(f"Cannot access target module {module_name}: {e}")

        if isinstance(module, nn.Linear):
            logger.debug(f"Converting {module_name} from nn.Linear to DisLoRALinear")

            try:
                dislora_module = DisLoRALinear(
                    in_features=module.weight.shape[0],
                    out_features=module.weight.shape[1],
                    r=dislora_config.r,
                    dislora_alpha=dislora_config.dislora_alpha,
                    dislora_dropout=dislora_config.dislora_dropout,
                    dash_flag=dislora_config.dash_flag,
                    s_tsd=dislora_config.s_tsd,
                    prefer_small_sigma=dislora_config.prefer_small_sigma,
                    merge_weights=dislora_config.merge_weights,
                    bias_attr=False if module.bias is None else None,
                    init_lora_weights=False,
                )

                dislora_module.weight.set_value(module.weight)
                if module.bias is not None:
                    dislora_module.bias.set_value(module.bias)

                dislora_module._init_lora_weights()

                setattr(parent_module, attribute_chain[-1], dislora_module)
                logger.debug(f"Successfully replaced {module_name}")

            except Exception as e:
                logger.error(f"Failed to create DisLoRALinear for {module_name}: {e}")
                raise ValueError(f"Failed to create DisLoRALinear for {module_name}: {e}")

        elif isinstance(module, DisLoRALinear):
            logger.debug(f"Module {module_name} is already a DisLoRALinear, skipping")

        else:

            module_type = type(module).__name__
            if any(keyword in module_name.lower() for keyword in ["dislora_dropout", "direc_"]):
                logger.debug(f"Skipping DisLoRA submodule {module_name} ({module_type})")
                return
            else:

                error_msg = f"Target module {module_name} is {module_type}, not nn.Linear. DisLoRA can only replace nn.Linear modules."
                logger.error(f"Cannot replace {module_name}: expected nn.Linear, got {module_type}")
                raise ValueError(error_msg)

    def _find_and_restore_module(self, module_name):
        parent_module = self.model
        attribute_chain = module_name.split(".")
        for name in attribute_chain[:-1]:
            parent_module = getattr(parent_module, name)
        module = getattr(parent_module, attribute_chain[-1])
        original_model_class = self.restore_layer_map[module.__class__]
        original_module = original_model_class(in_features=module.weight.shape[0], out_features=module.weight.shape[1])
        original_module.weight = module.weight

        if isinstance(module, DisLoRALinear):
            if not module.merged:
                complete_weight = module.weight + module.get_delta_weight()
                original_module.weight.set_value(complete_weight)
            else:
                original_module.weight.set_value(module.weight)
        else:
            original_module.weight.set_value(module.weight)

        if module.bias is not None:
            original_module.bias.set_value(module.bias)

        setattr(parent_module, attribute_chain[-1], original_module)

    def get_trainable_state_dict(self):
        """
        Obtain the required state dictionary to be saved, including:
        1. Trainable parameters (stop_gradient = False)
        2. Main weight W_prin (although frozen, must be saved)
        3. TSD direction parameters (although frozen, must be saved)
        4. QAT-related parameters
        """
        trainable_state_dict = OrderedDict()
        for name, weight in self.model.state_dict().items():
            # Save trainable parameters and QAT parameters
            if not weight.stop_gradient or "activation_quanter" in name or "weight_quanter" in name:
                trainable_state_dict[name] = weight
            # Save the main branch weight W_prin (for critical fixes)
            elif "weight" in name and any(layer_name in name for layer_name in [".weight"]) and "Direc_" not in name:
                trainable_state_dict[name] = weight
                logger.debug(f"Saving backbone weight: {name}")
            # Save all TSD parameters (excluding Direc_Stsd)
            elif any(tsd_param in name for tsd_param in ["Direc_Utsd", "Direc_Vhtsd"]):
                trainable_state_dict[name] = weight
                logger.debug(f"Saving TSD parameter: {name}")
            # Save the bias parameters (if any)
            elif "bias" in name and "Direc_" not in name:
                trainable_state_dict[name] = weight
                logger.debug(f"Saving bias parameter: {name}")

        return trainable_state_dict

    def print_trainable_parameters(self) -> None:
        freeze_numel = 0
        trainable_numel = 0
        for _, weight in self.model.state_dict().items():
            if weight.stop_gradient:
                freeze_numel += np.prod(weight.shape)
            else:
                trainable_numel += np.prod(weight.shape)
        logger.debug(
            f"Frozen parameters: {freeze_numel:.2e} || Trainable parameters:{trainable_numel:.2e} || Total parameters:{freeze_numel+trainable_numel:.2e}|| Trainable:{trainable_numel / (freeze_numel+trainable_numel):.2%}"
        )

    def mark_only_dislora_as_trainable(self) -> None:
        """
        Mark only the parameters related to DisLoRA as trainable, while ensuring that the TSD parameters remain in a frozen state.
        """

        for full_param_name, weight in self.model.state_dict().items():

            is_dislora_layer = any(
                re.fullmatch(target_module, full_param_name.rsplit(".", 1)[0])
                for target_module in self.dislora_config.target_modules
            )

            if is_dislora_layer:
                param_name = full_param_name.split(".")[-1]

                if param_name == "weight" and "Direc_" not in full_param_name:
                    weight.stop_gradient = True
                    logger.debug(f"Freezing backbone weight: {full_param_name}")

                elif param_name == "bias" and "Direc_" not in full_param_name:
                    if self.dislora_config.trainable_bias in ["dislora", "all"]:
                        weight.stop_gradient = False
                        logger.debug(f"Setting bias as trainable: {full_param_name}")
                    else:
                        weight.stop_gradient = True
                        logger.debug(f"Freezing bias: {full_param_name}")

                elif any(tsd_param in full_param_name for tsd_param in ["Direc_Utsd", "Direc_Vhtsd"]):
                    weight.stop_gradient = True
                    logger.debug(f"Keeping TSD parameter frozen: {full_param_name}")

                elif any(
                    trainable_param in full_param_name
                    for trainable_param in ["Direc_Ur", "Direc_Sr", "Direc_Vhr", "Direc_Stsd"]
                ):
                    weight.stop_gradient = False
                    logger.debug(f"Setting DisLoRA parameter as trainable: {full_param_name}")

                else:
                    weight.stop_gradient = True
                    logger.debug(f"Freezing other parameter: {full_param_name}")

            else:
                param_name = full_param_name.split(".")[-1]
                if self.dislora_config.trainable_bias == "all" and param_name == "bias":
                    weight.stop_gradient = False
                    logger.debug(f"Setting bias as trainable in non-DisLoRA layer: {full_param_name}")
                else:
                    weight.stop_gradient = True
                    logger.debug(f"Freezing parameter in non-DisLoRA layer: {full_param_name}")

        if self.dislora_config.trainable_modules is not None:
            for full_param_name, weight in self.model.state_dict().items():
                if any(
                    re.fullmatch(trainable_module, full_param_name)
                    for trainable_module in self.dislora_config.trainable_modules
                ):

                    if not any(tsd_param in full_param_name for tsd_param in ["Direc_Utsd", "Direc_Vhtsd"]):
                        weight.stop_gradient = False
                        logger.debug(f"Setting additional trainable module parameter: {full_param_name}")
                    else:
                        logger.warning(
                            f"TSD parameter {full_param_name} matched trainable_modules pattern but kept frozen"
                        )

    def get_dislora_model(self, model: Union[PretrainedModel, nn.Layer], dislora_config: DisLoRAConfig):
        """
        Iterate all base model layers, change target modules to DisLoRALayer.
        """
        if dislora_config.target_modules is None:
            return model
        else:
            target_modules = dislora_config.target_modules

        target_module_names = []

        existing_dislora_paths = set()
        for module_name, module in model.named_sublayers():
            if isinstance(module, DisLoRALinear):
                existing_dislora_paths.add(module_name)

        for target_module in target_modules:
            for module_name, module in model.named_sublayers():

                if re.fullmatch(target_module, module_name):

                    if not isinstance(module, DisLoRALinear):

                        is_submodule = any(
                            module_name.startswith(dislora_path + ".") for dislora_path in existing_dislora_paths
                        )

                        if not is_submodule:
                            target_module_names.append(module_name)
                        else:
                            logger.debug(f"Skipping {module_name} - it's a submodule of existing DisLoRA module")
                    else:
                        logger.debug(f"Skipping {module_name} - already a DisLoRA module")

        for module_name in target_module_names:
            try:
                self._find_and_replace_module(model, module_name, dislora_config)
                logger.debug(f"Replaced {module_name} with DisLoRALinear")
            except ValueError as e:
                raise e
            except Exception as e:

                logger.warning(f"Failed to replace {module_name}: {e}")

        return model

    def restore_original_model(self):
        # make sure W and dislora weights are not merged before we restore the original model
        for layer_name, layer in self.model.named_sublayers():
            if isinstance(layer, DisLoRALinear):
                self._find_and_restore_module(layer_name)
        return self.model

    def __getattr__(self, name: str):
        """
        Forward missing attributes to the wrapped module.
        """
        try:
            return super().__getattr__(name)  # defer to nn.Layer's logic
        except AttributeError:
            return getattr(self.model, name)

    def train(self):
        self.training = True
        self.model.training = True
        for layer in self.model.sublayers():
            layer.training = True
            layer.train()

    def eval(self):
        self.training = False
        self.model.training = False
        for layer in self.model.sublayers():
            layer.training = False
            layer.eval()

    def disable_dislora(self):
        """
        Disable the DisLoRA adapter
        """
        for _, layer in self.model.named_sublayers():
            if isinstance(layer, DisLoRALinear):
                layer.disable_adapters()

    def enable_dislora(self):
        """
        Enable the DisLoRA adapter
        """
        for _, layer in self.model.named_sublayers():
            if isinstance(layer, DisLoRALinear):
                layer.enable_adapters()

    def merge(self):
        for _, layer in self.model.named_sublayers():
            if any(isinstance(layer, dislora_layer) for dislora_layer in AVAILABLE_LAYERS):
                layer.merge()

    def unmerge(self):
        for _, layer in self.model.named_sublayers():
            if any(isinstance(layer, dislora_layer) for dislora_layer in AVAILABLE_LAYERS):
                layer.unmerge()

    def get_model_config(
        self,
    ):
        return self.model_config.to_dict()
