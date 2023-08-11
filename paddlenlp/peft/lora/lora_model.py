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

import math
import os
import re
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Union

import paddle
import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)

from ...transformers.conversion_utils import ConversionMixin
from ...transformers.model_utils import PretrainedModel, _add_variant, dtype_guard
from ...utils.distributed import distributed_gather
from ...utils.env import LORA_WEIGHTS_NAME
from ...utils.log import logger
from .lora_config import LoRAConfig
from .lora_layers import (
    ColumnParallelLoRALinear,
    ColumnParallelLoRAMergedLinear,
    LoRALinear,
    LoRAMergedLinear,
    RowParallelLoRALinear,
)


class LoRAModel(nn.Layer):
    restore_layer_map: Dict[nn.Layer, nn.Layer] = {
        LoRALinear: nn.Linear,
        LoRAMergedLinear: nn.Linear,
        ColumnParallelLoRALinear: ColumnParallelLinear,
        ColumnParallelLoRAMergedLinear: ColumnParallelLinear,
    }

    def __init__(self, model, lora_config: LoRAConfig) -> None:
        super().__init__()
        self.lora_config = lora_config
        self.lora_split_mapping = {}
        if self.lora_config.dtype is None:
            self.lora_config.dtype = paddle.get_default_dtype()
        with dtype_guard(self.lora_config.dtype):
            self.model = self.get_lora_model(model, lora_config)
        if self.lora_config.tensor_parallel_degree != self.model.config.tensor_parallel_degree:
            self.lora_config.tensor_parallel_degree = self.model.config.tensor_parallel_degree
            logger.warning(
                f"Reset tensor_parallel_degree of lora_config to {self.model.config.tensor_parallel_degree}."
            )
        self.forward = self.model.forward

    def add_lora_split_mapping(self, module_name, is_column=False):
        self.lora_split_mapping[module_name] = is_column

    def _get_tensor_parallel_mappings(self, config, is_split=True):

        from paddlenlp.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings():
            final_actions = {}
            for key, is_col in self.lora_split_mapping.items():
                final_actions[key] = partial(fn, is_column=is_col)

            return final_actions

        mappings = get_tensor_parallel_split_mappings()

        return mappings

    @classmethod
    def from_pretrained(cls, model, lora_path, **kwargs):
        lora_config = kwargs.pop("lora_config", None)
        # init lora config & lora model
        if not isinstance(lora_config, LoRAConfig):
            lora_config = LoRAConfig.from_pretrained(lora_path)
        # define a new variable to conserve original lora_config.tensor_parallel_degree value which will update while initializing lora model
        lora_config_tensor_parallel_degree = lora_config.tensor_parallel_degree
        lora_model = cls(model, lora_config)

        # define lora weight name
        if lora_config_tensor_parallel_degree > 1:
            lora_weight_name = _add_variant(LORA_WEIGHTS_NAME, f"tp{model.config.tensor_parallel_rank:0>2d}")
        else:
            lora_weight_name = LORA_WEIGHTS_NAME

        # load and set lora weight parameter
        lora_weight_path = os.path.join(lora_path, lora_weight_name)
        if os.path.exists(lora_weight_path):
            # load lora weight parameter
            lora_state_dict = paddle.load(lora_weight_path, return_numpy=True)
            logger.info(f"Loading the LoRA weights from {lora_weight_path}")

            if (
                lora_config_tensor_parallel_degree > 1
                and lora_config_tensor_parallel_degree != model.config.tensor_parallel_degree
            ):
                raise NotImplementedError(
                    f"{lora_config_tensor_parallel_degree} is not equal to {model.config.tensor_parallel_degree}. Please merge LoRA weights first."
                )

            # convert parameters to tensor parallel for mp model
            if lora_config_tensor_parallel_degree <= 1 and model.config.tensor_parallel_degree > 1:
                lora_state_dict = lora_model._convert_tensor_parallel(lora_state_dict=lora_state_dict)

            # set lora state dict
            lora_model.set_state_dict(lora_state_dict)
        else:
            logger.error(f"LoRA weights not found under {lora_path}, creating LoRA weights from scratch")

        return lora_model

    def set_state_dict(self, state_dict):
        import warnings

        warnings.filterwarnings(
            action="ignore", message=".*Skip loading for.*", category=Warning, lineno=0, append=False
        )
        self.model.set_state_dict(state_dict)
        logger.info("Load lora weight successfully")

    def _merge_trainable_tensor_parallel(self, trainable_state_dict):
        trainable_name_action_mappings = self._get_tensor_parallel_mappings(self.model.config, is_split=False)

        name_action_mappings = self.model._get_tensor_parallel_mappings(self.model.config, is_split=False)
        state_keys_map = ConversionMixin._resolve_prefix_keys(
            name_action_mappings.keys(), self.model.state_dict().keys()
        )
        for k, v in state_keys_map.items():
            if v in trainable_state_dict:
                trainable_name_action_mappings[v] = name_action_mappings[k]

        hcg = paddle.distributed.fleet.get_hybrid_communicate_group()
        mp_group = hcg.get_model_parallel_group()
        is_dst = paddle.distributed.get_rank(mp_group) == 0

        for key in trainable_state_dict:
            tensor = trainable_state_dict[key]
            if key in trainable_name_action_mappings:
                ret = distributed_gather(tensor, group=mp_group, offload=True)
                action = trainable_name_action_mappings[key]
                tensor = action(ret) if is_dst else None
                trainable_state_dict[key] = tensor
            else:
                trainable_state_dict[key] = tensor.numpy() if is_dst else None

        return trainable_state_dict

    def _convert_tensor_parallel(self, lora_state_dict):
        lora_name_action_mappings = self._get_tensor_parallel_mappings(self.model.config, is_split=False)

        name_action_mappings = self.model._get_tensor_parallel_mappings(self.model.config, is_split=False)
        state_keys_map = ConversionMixin._resolve_prefix_keys(
            name_action_mappings.keys(), self.model.state_dict().keys()
        )
        for k, v in state_keys_map.items():
            if v in lora_state_dict.keys():
                lora_name_action_mappings[v] = name_action_mappings[k]

        for name, action in lora_name_action_mappings.items():
            tensor = lora_state_dict.pop(name)
            lora_state_dict[name] = action(tensor)
        return lora_state_dict

    def save_pretrained(self, save_directory: str, merge_tensor_parallel: bool = True, **kwargs):
        variant = kwargs.get("variant", None)
        is_main_process = kwargs.get("is_main_process", paddle.distributed.get_rank() == 0)

        assert not os.path.isfile(
            save_directory
        ), f"Saving directory ({save_directory}) should be a directory, not a file"
        os.makedirs(save_directory, exist_ok=True)

        if merge_tensor_parallel and self.model.config.tensor_parallel_degree > 1:
            trainable_state_dict = self.get_trainable_state_dict()
            trainable_state_dict = self._merge_trainable_tensor_parallel(trainable_state_dict)
            if not is_main_process:
                logger.info("Saving with merge_tensor_parallel, tensor_parallel_rank > 0 don't need save")
                return
            variant = None
            self.lora_config.tensor_parallel_degree = -1
        else:
            trainable_state_dict = self.get_trainable_state_dict()
            if self.model.config.tensor_parallel_degree > 1:
                if variant is None:
                    variant = f"tp{self.model.config.tensor_parallel_rank:0>2d}"

        # save lora weight
        lora_weight_name = _add_variant(LORA_WEIGHTS_NAME, variant)
        weight_filename = os.path.join(save_directory, lora_weight_name)
        paddle.save(trainable_state_dict, weight_filename)

        # save lora config
        if is_main_process:
            self.lora_config.save_pretrained(save_directory)
        self.lora_config.tensor_parallel_degree = self.model.config.tensor_parallel_degree

    def _find_and_replace_module(self, model, module_name, lora_config, enable_lora):
        parent_module = model
        attribute_chain = module_name.split(".")
        for name in attribute_chain[:-1]:
            parent_module = getattr(parent_module, name)
        module = getattr(parent_module, attribute_chain[-1])
        lora_module = None
        if enable_lora is None:
            if isinstance(module, nn.Linear):
                lora_module = LoRALinear(
                    in_features=module.weight.shape[0],
                    out_features=module.weight.shape[1],
                    r=lora_config.r,
                    lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout,
                    merge_weights=lora_config.merge_weights,
                )
            elif isinstance(module, ColumnParallelLinear):
                # recover the original output_features
                output_features = module.weight.shape[1] * module.world_size
                lora_module = ColumnParallelLoRALinear(
                    in_features=module.weight.shape[0],
                    out_features=output_features,
                    gather_output=module.gather_output,
                    has_bias=module.bias is not None,
                    r=lora_config.r,
                    lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout,
                    merge_weights=lora_config.merge_weights,
                    lora_A_weight_attr=paddle.ParamAttr(
                        initializer=nn.initializer.KaimingUniform(
                            negative_slope=math.sqrt(5), nonlinearity="leaky_relu"
                        )
                    ),
                )
                # Lora column parallel will spilt lora B matrix
                self.add_lora_split_mapping(module_name + ".lora_B", is_column=True)
            elif isinstance(module, RowParallelLinear):
                # recover the original output_features
                lora_module = RowParallelLoRALinear(
                    in_features=module.weight.shape[0] * module.world_size,
                    out_features=module.weight.shape[1],
                    has_bias=module.bias is not None,
                    input_is_parallel=module.input_is_parallel,
                    r=lora_config.r,
                    lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout,
                    merge_weights=lora_config.merge_weights,
                )
                # Lora column parallel will spilt lora A matrix
                self.add_lora_split_mapping(module_name + ".lora_A", is_column=False)
        else:
            if isinstance(module, nn.Linear):
                lora_module = LoRAMergedLinear(
                    in_features=module.weight.shape[0],
                    out_features=module.weight.shape[1],
                    r=lora_config.r,
                    lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout,
                    merge_weights=lora_config.merge_weights,
                    enable_lora=enable_lora,
                    head_dim=lora_config.head_dim,
                )
            elif isinstance(module, ColumnParallelLinear):
                # recover the original output_features
                lora_module = ColumnParallelLoRAMergedLinear(
                    in_features=module.weight.shape[0],
                    out_features=module.weight.shape[1] * module.world_size,
                    gather_output=module.gather_output,
                    has_bias=module.bias is not None,
                    r=lora_config.r,
                    lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout,
                    merge_weights=lora_config.merge_weights,
                    enable_lora=enable_lora,
                    head_dim=lora_config.head_dim,
                    lora_A_weight_attr=paddle.ParamAttr(
                        initializer=nn.initializer.KaimingUniform(
                            negative_slope=math.sqrt(5), nonlinearity="leaky_relu"
                        )
                    ),
                )
        if lora_module is None:
            raise ValueError(
                f"LoRA strategy only supports paddle.nn.Linear or paddle.distributed.fleet.meta_parallel.ColumnParallelLinear. {module}({module_name}) is not supported。"
            )

        lora_module.weight = module.weight
        if module.bias is not None:
            lora_module.bias = module.bias
        setattr(parent_module, attribute_chain[-1], lora_module)

    def _find_and_restore_module(self, module_name):
        parent_module = self.model
        attribute_chain = module_name.split(".")
        for name in attribute_chain[:-1]:
            parent_module = getattr(parent_module, name)
        module = getattr(parent_module, attribute_chain[-1])
        original_model_class = self.restore_layer_map[module.__class__]
        original_module = original_model_class(in_features=module.weight.shape[0], out_features=module.weight.shape[1])
        original_module.weight = module.weight
        if module.bias is not None:
            original_module.bias = module.bias
        setattr(parent_module, attribute_chain[-1], original_module)

    def get_trainable_state_dict(self):
        trainable_state_dict = OrderedDict()
        for name, weight in self.model.state_dict().items():
            # get lora parameter & QAT scale parameter
            if not weight.stop_gradient or "activation_quanter" in name or "weight_quanter" in name:
                trainable_state_dict[name] = weight
        return trainable_state_dict

    def print_trainable_parameters(self) -> None:
        freeze_numel = 0
        trainable_numel = 0
        for _, weight in self.model.state_dict().items():
            if weight.stop_gradient:
                freeze_numel += weight.numel().item()
            else:
                trainable_numel += weight.numel().item()
        logger.info(
            f"Frozen parameters: {freeze_numel:.2e} || Trainable parameters:{trainable_numel:.2e} || Total parameters:{freeze_numel+trainable_numel:.2e}|| Trainable:{trainable_numel / (freeze_numel+trainable_numel):.2%}"
        )

    def mark_only_lora_as_trainable(self) -> None:
        for _, layer in self.model.named_sublayers():
            if (
                isinstance(layer, LoRALinear)
                or isinstance(layer, ColumnParallelLoRALinear)
                or isinstance(layer, RowParallelLoRALinear)
                or isinstance(layer, LoRAMergedLinear)
                or isinstance(layer, ColumnParallelLoRAMergedLinear)
            ):
                for name, weight in layer.state_dict().items():
                    if self.lora_config.trainable_bias in ["lora", "all"] and "bias" in name:
                        weight.stop_gradient = False
                    elif "lora" in name:
                        weight.stop_gradient = False
                    else:
                        weight.stop_gradient = True
            else:
                for name, weight in layer.state_dict().items():
                    if self.lora_config.trainable_bias == "all" and "bias" in name:
                        weight.stop_gradient = False
                    else:
                        weight.stop_gradient = True
        if self.lora_config.trainable_modules is not None:
            for name, weight in self.model.state_dict().items():
                if any(
                    re.fullmatch(trainable_module, name) for trainable_module in self.lora_config.trainable_modules
                ):
                    weight.stop_gradient = False

    def get_lora_model(self, model: Union[PretrainedModel, nn.Layer], lora_config: LoRAConfig):

        if lora_config.target_modules is None:
            return model
        elif isinstance(lora_config.target_modules, str):
            target_modules = [lora_config.target_modules]
            if lora_config.enable_lora_list is None or (
                isinstance(lora_config.enable_lora_list, List)
                and all(isinstance(item, bool) for item in lora_config.enable_lora_list)
            ):
                enable_lora_list = [lora_config.enable_lora_list]
            else:
                raise TypeError(
                    f"Invalid `enable_lora_list` value: {lora_config.enable_lora_list}. Since `target_modules` is `str`, `enable_lora_list` must be `None` or `List[bool]`"
                )
        else:
            target_modules = lora_config.target_modules
            if lora_config.enable_lora_list is None:
                enable_lora_list = [None for _ in range(len(target_modules))]
            elif isinstance(lora_config.enable_lora_list, List):
                enable_lora_list = lora_config.enable_lora_list
                if len(enable_lora_list) != len(target_modules):
                    raise TypeError(
                        f"Invalid lora_config.enable_lora_list value: {lora_config.enable_lora_list}. Since lora_config.target_modules is `List[str]`, `enable_lora_list` should have the same length as `target_modules`"
                    )
                for enable_lora in enable_lora_list:
                    if not (
                        enable_lora is None
                        or (isinstance(enable_lora, List) and all(isinstance(item, bool) for item in enable_lora))
                    ):
                        raise TypeError(
                            f"Invalid `enable_lora_list` value: {lora_config.enable_lora_list}. Since `target_modules` is `List[str]`, `enable_lora_list` must be `None` or  `List[Optional[List[bool]]]`"
                        )
            else:
                raise TypeError(
                    f"Invalid `enable_lora_list` value: {lora_config.enable_lora_list}. Since `target_modules` is `List[str]`, `enable_lora_list` must be `None` or `List[Optional[List[bool]]]`"
                )

        for target_module, enable_lora in zip(target_modules, enable_lora_list):
            for i in model.named_sublayers():
                module_name = i[0]
                if re.fullmatch(target_module, module_name):
                    self._find_and_replace_module(model, module_name, lora_config, enable_lora)
        return model

    def restore_original_model(self):
        # make sure W and lora weights are not merged before we restore the original model
        if self.lora_config.merge_weights:
            self.train()

        for layer_name, layer in self.model.named_sublayers():
            if (
                isinstance(layer, LoRALinear)
                or isinstance(layer, ColumnParallelLoRALinear)
                or isinstance(layer, LoRAMergedLinear)
                or isinstance(layer, ColumnParallelLoRAMergedLinear)
            ):
                self._find_and_restore_module(layer_name)
        return self.model

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
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
