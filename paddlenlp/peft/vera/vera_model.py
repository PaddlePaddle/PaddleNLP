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
import math
import os
import re
import tempfile
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Union


import aistudio_sdk
import numpy as np
import paddle
import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    PipelineLayer,
    RowParallelLinear,
)

from ...transformers.conversion_utils import ConversionMixin
from ...transformers.model_utils import PretrainedModel, _add_variant, dtype_guard
from ...transformers.utils import weight_name_suffix
from ...utils.distributed import distributed_gather
from ...utils.env import VERA_WEIGHTS_NAME
from ...utils.log import logger

from .vera_config import VeRAConfig
from .vera_layers import VeRALinear



class VeRAModel(nn.Layer):
    restore_layer_map: Dict[nn.Layer, nn.Layer] = {
        VeRALinear: nn.Linear,
    }

    def __init__(self, model, vera_config: VeRAConfig) -> None:
        super().__init__()
        self.quantized = False
        self.vera_config = vera_config
        self.lora_split_mapping = {}
        if self.vera_config.dtype is None:
            self.vera_config.dtype = paddle.get_default_dtype()
        with dtype_guard(self.vera_config.dtype):
            self.model = self.get_vera_model(model, vera_config)
        self.is_pipelinemodel = False
        if issubclass(type(self.model), PipelineLayer):
            self.is_pipelinemodel = True
            self.model._single_to_pp_mapping = None
        if self.vera_config.tensor_parallel_degree != self.model.config.tensor_parallel_degree:
            self.vera_config.tensor_parallel_degree = self.model.config.tensor_parallel_degree
            logger.warning(
                f"Reset tensor_parallel_degree of vera_config to {self.model.config.tensor_parallel_degree}."
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
    def from_pretrained(cls, model, vera_path, **kwargs):
        vera_config = kwargs.pop("vera_config", None)
        # init lora config & lora model
        if not isinstance(vera_config, VeRAConfig):
            vera_config = VeRAConfig.from_pretrained(vera_path)
        # define a new variable to conserve original vera_config.tensor_parallel_degree value which will update while initializing lora model
        vera_config_tensor_parallel_degree = vera_config.tensor_parallel_degree
        lora_model = cls(model, vera_config)
        
        print('cls vera model', lora_model)

        # define lora weight name
        if vera_config_tensor_parallel_degree > 1:
            lora_weight_name = _add_variant(VERA_WEIGHTS_NAME, f"tp{model.config.tensor_parallel_rank:0>2d}")
        else:
            lora_weight_name = VERA_WEIGHTS_NAME

        # load and set lora weight parameter
        lora_weight_path = os.path.join(vera_path, lora_weight_name)
        print('lora_weight_path', lora_weight_path)
        if os.path.exists(lora_weight_path):
            # load lora weight parameter
            print('lora_weight_path existed, loading lora weight parameter')
            
            lora_state_dict = paddle.load(lora_weight_path, return_numpy=True)
            logger.info(f"Loading the LoRA weights from {lora_weight_path}")

            if (
                vera_config_tensor_parallel_degree > 1
                and vera_config_tensor_parallel_degree != model.config.tensor_parallel_degree
            ):
                raise NotImplementedError(
                    f"{vera_config_tensor_parallel_degree} is not equal to {model.config.tensor_parallel_degree}. Please merge LoRA weights first."
                )

            # convert parameters to tensor parallel for mp model
            if vera_config_tensor_parallel_degree <= 1 and model.config.tensor_parallel_degree > 1:
                lora_state_dict = lora_model._convert_tensor_parallel(lora_state_dict=lora_state_dict)

            # set lora state dict
            lora_model.set_state_dict(lora_state_dict)
        else:
            logger.error(f"LoRA weights not found under {vera_path}, creating LoRA weights from scratch")

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
                is_collumn = self.lora_split_mapping[key]
                if "_scale" in key and not is_collumn and is_dst:
                    ret = paddle.to_tensor(ret)
                    tensor = paddle.max(ret, axis=0)
                else:
                    tensor = action(ret) if is_dst else None
                trainable_state_dict[key] = tensor
            else:
                trainable_state_dict[key] = tensor.cpu().numpy() if is_dst else None

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
            if name in lora_state_dict:
                tensor = lora_state_dict.pop(name)
                lora_state_dict[name] = action(tensor)
            else:
                logger.warning(f"{name} not found in lora_state_dict!")
        return lora_state_dict

    def save_pretrained(self, save_directory: str, merge_tensor_parallel: bool = False, **kwargs):
        
        print('save vera pretrained')
        save_model_config = kwargs.get("save_model_config", True)

        if self.is_pipelinemodel:
            self.model._single_to_pp_mapping = None
        if self.quantized and merge_tensor_parallel and self.vera_config.tensor_parallel_degree > 1:
            merge_tensor_parallel = False
            logger.warning(
                "Quantized strategy does not support merge_tensor_parallel. Set merge_tensor_parallel to False."
            )
        if self.is_pipelinemodel and merge_tensor_parallel and self.vera_config.tensor_parallel_degree > 1:
            merge_tensor_parallel = False
            logger.warning(
                "Pipeline parallism does not support merge_tensor_parallel. Set merge_tensor_parallel to False."
            )

        variant = kwargs.get("variant", None)
        is_main_process = kwargs.get("is_main_process", paddle.distributed.get_rank() == 0)

        assert not os.path.isfile(
            save_directory
        ), f"Saving directory ({save_directory}) should be a directory, not a file"
        os.makedirs(save_directory, exist_ok=True)

        vera_config_to_save = VeRAConfig(**self.vera_config.to_dict())
        
        print('vera_config_to_save', vera_config_to_save)

        if merge_tensor_parallel and vera_config_to_save.tensor_parallel_degree > 1:
            trainable_state_dict = self.get_trainable_state_dict()
            trainable_state_dict = self._merge_trainable_tensor_parallel(trainable_state_dict)
            if not is_main_process:
                logger.info("Saving with merge_tensor_parallel, tensor_parallel_rank > 0 don't need save")
                return
            if variant is not None and "tp" in variant:
                variant = "_".join([x for x in variant.split("_") if "tp" not in x])
            vera_config_to_save.tensor_parallel_degree = -1
        else:
            trainable_state_dict = self.get_trainable_state_dict()
            if vera_config_to_save.tensor_parallel_degree > 1:
                if variant is None:
                    variant = weight_name_suffix()

        # save lora weight
        lora_weight_name = _add_variant(VERA_WEIGHTS_NAME, variant)
        weight_filename = os.path.join(save_directory, lora_weight_name)
        paddle.save(trainable_state_dict, weight_filename)

        # save lora config
        if is_main_process:
            vera_config_to_save.save_pretrained(save_directory)
            if save_model_config:
                model_config_to_save = copy.deepcopy(self.model.config)
                if merge_tensor_parallel:
                    model_config_to_save.tensor_parallel_degree = -1
                model_config_to_save.save_pretrained(save_directory)

    def _find_and_replace_module(self, model, module_name, vera_config, enable_lora):
        parent_module = model
        attribute_chain = module_name.split(".")
        for name in attribute_chain[:-1]:
            parent_module = getattr(parent_module, name)
        module = getattr(parent_module, attribute_chain[-1])
        lora_module = None
        if enable_lora is None:
            if isinstance(module, nn.Linear):
                # print(f'module:{module}',type(module))
                # print(f'parent module:{parent_module}')
                # print(f'parent_module name:{module_name}')
                # print(f'module.weight:', module.weight)
                # print(f'vera_config.merge_weights', vera_config.merge_weights)
                # exit(0)
                lora_module = VeRALinear(
                    # 将要替换的层传递过去
                    base_linear_module=module,
                    in_features=module.weight.shape[0],
                    out_features=module.weight.shape[1],
                    r=vera_config.r,
                    lora_alpha=vera_config.lora_alpha,
                    lora_dropout=vera_config.lora_dropout,
                    merge_weights=vera_config.merge_weights,
                    bias_attr=False if module.bias is None else None,
                    pissa_init=vera_config.pissa_init
                )
 


        if lora_module is None:
            raise ValueError(
                f"LoRA strategy only supports paddle.nn.Linear or paddle.distributed.fleet.meta_parallel.ColumnParallelLinear. {module}({module_name}) is not supported。"
            )
        if getattr(lora_module, "quant_weight", None) is not None:
            lora_module.quant_weight = module.quant_weight
            if getattr(lora_module, "quant_scale", None) is not None:
                lora_module.quant_scale = module.quant_scale
            if getattr(lora_module, "qquant_scale", None) is not None:
                lora_module.qquant_scale = module.qquant_scale
            if getattr(lora_module, "double_quant_scale", None) is not None:
                lora_module.double_quant_scale = module.double_quant_scale
            if getattr(lora_module, "quant_sacle_offset", None) is not None:
                lora_module.quant_sacle_offset = module.quant_sacle_offset
        # else:
        #     lora_module.weight = module.weight
        #     print('self weight', lora_module.weight)
        #     exit(0)
        if module.bias is not None:
            lora_module.bias = module.bias
            
        # print('lora module', lora_module)
        # print('bias', lora_module.bias)
        # print('lora module', lora_module.weight)
        # print('lora_module.lora_A', lora_module.lora_A)
        # exit(0)
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
                freeze_numel += np.prod(weight.shape)
            else:
                trainable_numel += np.prod(weight.shape)
        logger.debug(
            f"Frozen parameters: {freeze_numel:.2e} || Trainable parameters:{trainable_numel:.2e} || Total parameters:{freeze_numel+trainable_numel:.2e}|| Trainable:{trainable_numel / (freeze_numel+trainable_numel):.2%}"
        )

    def mark_only_vera_as_trainable(self, notfreezeB=False) -> None:
        for _, layer in self.model.named_sublayers():
            if isinstance(layer, VeRALinear):
                for name, weight in layer.state_dict().items():
                    if self.vera_config.trainable_bias in ["lora", "all"] and "bias" in name:
                        weight.stop_gradient = False
                    elif "lora" in name or 'vera' in name:
                        # freezeB=True, vera_b, vera_d, lora_B可训练
                        # freezeB=False, vera_b, vera_d 可训练
                        if 'vera' in name:
                            weight.stop_gradient = False
                        elif 'lora_B' in name and notfreezeB:
                            weight.stop_gradient = False
                        else:
                            weight.stop_gradient = True
                    else:
                        weight.stop_gradient = True
            else:
                for name, weight in layer.state_dict().items():
                    if self.vera_config.trainable_bias == "all" and "bias" in name:
                        weight.stop_gradient = False
                    else:
                        weight.stop_gradient = True
        if self.vera_config.trainable_modules is not None:
            for name, weight in self.model.state_dict().items():
                if any(
                    re.fullmatch(trainable_module, name) for trainable_module in self.vera_config.trainable_modules
                ):
                    weight.stop_gradient = False

    def get_vera_model(self, model: Union[PretrainedModel, nn.Layer], vera_config: VeRAConfig):

        if vera_config.target_modules is None:
            return model
        elif isinstance(vera_config.target_modules, str):
            target_modules = [vera_config.target_modules]
            if vera_config.enable_lora_list is None or (
                isinstance(vera_config.enable_lora_list, List)
                and all(isinstance(item, bool) for item in vera_config.enable_lora_list)
            ):
                enable_lora_list = [vera_config.enable_lora_list]
            else:
                raise TypeError(
                    f"Invalid `enable_lora_list` value: {vera_config.enable_lora_list}. Since `target_modules` is `str`, `enable_lora_list` must be `None` or `List[bool]`"
                )
        else:
            target_modules = vera_config.target_modules
            if vera_config.enable_lora_list is None:
                enable_lora_list = [None for _ in range(len(target_modules))]
            elif isinstance(vera_config.enable_lora_list, List):
                enable_lora_list = vera_config.enable_lora_list
                if len(enable_lora_list) != len(target_modules):
                    raise TypeError(
                        f"Invalid vera_config.enable_lora_list value: {vera_config.enable_lora_list}. Since vera_config.target_modules is `List[str]`, `enable_lora_list` should have the same length as `target_modules`"
                    )
                for enable_lora in enable_lora_list:
                    if not (
                        enable_lora is None
                        or (isinstance(enable_lora, List) and all(isinstance(item, bool) for item in enable_lora))
                    ):
                        raise TypeError(
                            f"Invalid `enable_lora_list` value: {vera_config.enable_lora_list}. Since `target_modules` is `List[str]`, `enable_lora_list` must be `None` or  `List[Optional[List[bool]]]`"
                        )
            else:
                raise TypeError(
                    f"Invalid `enable_lora_list` value: {vera_config.enable_lora_list}. Since `target_modules` is `List[str]`, `enable_lora_list` must be `None` or `List[Optional[List[bool]]]`"
                )

        for target_module, enable_lora in zip(target_modules, enable_lora_list):
            for i in model.named_sublayers():
                module_name = i[0]
                if re.fullmatch(target_module, module_name):
                    self._find_and_replace_module(model, module_name, vera_config, enable_lora)
        return model

    def restore_original_model(self):
        # make sure W and lora weights are not merged before we restore the original model
        if self.vera_config.merge_weights:
            self.train()

        for layer_name, layer in self.model.named_sublayers():
            if isinstance(layer, VeRALinear):
                self._find_and_restore_module(layer_name)
            else:
                raise NotImplementedError(f"{layer} restoration is not supported yet.")
        return self.model

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        # print('-'* 100)
        # print('_getattr__', name)
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

    def save_to_aistudio(
        self,
        repo_id,
        private=True,
        license="Apache License 2.0",
        exist_ok=True,
        subfolder=None,
        merge_tensor_parallel=False,
        **kwargs
    ):
        """
        Uploads all elements of this model to a new AiStudio Hub repository.
        Args:
            repo_id (str): Repository name for your model/tokenizer in the Hub.
            token (str): Your token for the Hub.
            private (bool, optional): Whether the model/tokenizer is set to private. Defaults to True.
            license (str): The license of your model/tokenizer. Defaults to: "Apache License 2.0".
            exist_ok (bool, optional): Whether to override existing repository. Defaults to: True.
            subfolder (str, optional): Push to a subfolder of the repo instead of the root
            merge_tensor_parallel (bool): Whether to merge the tensor parallel weights. Defaults to False.
        """
        res = aistudio_sdk.hub.create_repo(repo_id=repo_id, private=private, license=license, **kwargs)
        if "error_code" in res:
            if res["error_code"] == 10003 and exist_ok:
                logger.info(
                    f"Repo {repo_id} already exists, it will override files with the same name. To avoid this, please set exist_ok=False"
                )
            else:
                logger.error(
                    f"Failed to create repo {repo_id}, error_code: {res['error_code']}, error_msg: {res['error_msg']}"
                )
        else:
            logger.info(f"Successfully created repo {repo_id}")

        with tempfile.TemporaryDirectory() as root_dir:
            if subfolder is not None:
                save_dir = os.path.join(root_dir, subfolder)
            else:
                save_dir = root_dir
            # save model
            self.save_pretrained(save_dir, merge_tensor_parallel=merge_tensor_parallel)

            # Upload model and return
            logger.info(f"Pushing to the {repo_id}. This might take a while")
            for filename in os.listdir(save_dir):
                res = aistudio_sdk.hub.upload(
                    repo_id=repo_id, path_or_fileobj=os.path.join(save_dir, filename), path_in_repo=filename, **kwargs
                )
                if "error_code" in res:
                    logger.error(
                        f"Failed to upload {filename}, error_code: {res['error_code']}, error_msg: {res['error_msg']}"
                    )
                else:
                    logger.info(f"{filename}: {res['message']}")
