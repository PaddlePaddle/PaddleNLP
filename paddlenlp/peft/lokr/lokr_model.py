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
import gc
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
from paddle.distributed.fleet.meta_parallel import PipelineLayer

from paddlenlp.transformers import AutoConfig, PretrainedModel
from paddlenlp.transformers.conversion_utils import ConversionMixin
from paddlenlp.transformers.model_utils import (
    _add_variant,
    _load_state_dict_into_model,
    dtype_guard,
    load_state_dict,
)
from paddlenlp.transformers.utils import get_checkpoint_shard_files, weight_name_suffix
from paddlenlp.utils.distributed import distributed_allgather, distributed_gather
from paddlenlp.utils.env import SAFE_PEFT_WEIGHTS_INDEX_NAME
from paddlenlp.utils.log import logger
from paddlenlp.utils.tools import get_env_device

from .lokr_config import LoKrConfig
from .lokr_envs import LOKR_WEIGHTS_NAME


def get_lokr_layers():
    from .lokr_layers import LoKrLinear

    return {
        "LoKrLinear": LoKrLinear,
    }


lokr_layers = get_lokr_layers()
LoKrLinear = lokr_layers["LoKrLinear"]
AVAILABLE_LAYERS = [
    LoKrLinear,
]
# try:
#     from paddlenlp.lora_quantization_layers import ( # type: ignore
#         ColumnParallelQuantizationLoRALinear,
#         QuantizationLoRALinear,
#         RowParallelQuantizationLoRALinear,
#     )

#     AVAILABLE_LAYERS += [
#         ColumnParallelQuantizationLoRALinear,
#         QuantizationLoRALinear,
#         RowParallelQuantizationLoRALinear,
#     ]
# except:
#     QuantizationLinear = None
#     ColumnParallelQuantizationLinear = None
#     RowParallelQuantizationLinear = None
#     QuantizationLoRALinear = None
#     ColumnParallelQuantizationLoRALinear = None
#     RowParallelQuantizationLoRALinear = None


class LoKrModel(nn.Layer):
    # TODO:lugimzzz support restore in following PR
    restore_layer_map: Dict[nn.Layer, nn.Layer] = {
        LoKrLinear: nn.Linear,
        # ColumnParallelLoRALinear: ColumnParallelLinear,
        # RowParallelLoRALinear: RowParallelLinear,
        # QuantizationLoRALinear: QuantizationLinear,
    }

    def __init__(self, model, lokr_config: LoKrConfig) -> None:
        super().__init__()
        self.model_config = AutoConfig.from_pretrained(lokr_config.base_model_name_or_path)
        self.quantized = False
        self.lokr_config = lokr_config
        self.lokr_split_mapping = {}
        if self.lokr_config.dtype is None:
            self.lokr_config.dtype = paddle.get_default_dtype()
        with dtype_guard(self.lokr_config.dtype):
            self.model = self.get_lokr_model(model, lokr_config)
        self.is_pipelinemodel = False
        if issubclass(type(self.model), PipelineLayer):
            self.is_pipelinemodel = True
            self.model._single_to_pp_mapping = None
        if self.lokr_config.tensor_parallel_degree != self.model.config.tensor_parallel_degree:
            self.lokr_config.tensor_parallel_degree = self.model.config.tensor_parallel_degree
            logger.warning(
                f"Reset tensor_parallel_degree of lokr_config to {self.model.config.tensor_parallel_degree}."
            )
        self.forward = self.model.forward

        logger.info("Mark only lokr and trainable_module as trainable.")
        self.mark_only_lokr_as_trainable()

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

        rename_lora_split_mapping = {}
        if issubclass(type(self.model), PipelineLayer):
            # rename lora_split_mapping
            prefixes = self.model.get_sequential_name_prefixes()
            keys = self.lora_split_mapping.keys()
            first_key = ""
            for k in keys:
                first_key = k
                break
            first_key = first_key.split(".")
            use_virtual_pp_degree = first_key[0].isdigit() and first_key[1].isdigit()

            for k in keys:
                name_splited = k.split(".")
                if use_virtual_pp_degree:
                    if name_splited[0].isdigit():
                        if name_splited[1].isdigit():
                            idx = str(int(name_splited[0]) + int(name_splited[1]))
                            single_name = [prefixes[idx]]
                            single_name.extend(name_splited[2:])
                        else:
                            single_name = [prefixes[str(len(prefixes) - 1)]]
                            single_name.extend(name_splited[2:])
                            logger.warning(
                                f"Please check! we treat this key as last layer, get {k}, set origin name as {'.'.join(single_name)}"
                            )
                    else:
                        raise ValueError(f"Please check! {k} is not a valid key.")
                else:
                    idx = name_splited[0]
                    # for normal pp layer name
                    if idx.isdigit():
                        single_name = [prefixes[idx]]
                        single_name.extend(name_splited[1:])
                    else:
                        raise ValueError(f"Unexpected key: {k} for pp lora layer.")
                rename_lora_split_mapping[".".join(single_name)] = self.lora_split_mapping[k]

        lora_split_mapping = (
            rename_lora_split_mapping if issubclass(type(self.model), PipelineLayer) else self.lora_split_mapping
        )

        def get_tensor_parallel_split_mappings():
            final_actions = {}
            for key, is_col in lora_split_mapping.items():
                final_actions[key] = partial(fn, is_column=is_col)

            return final_actions

        mappings = get_tensor_parallel_split_mappings()

        return mappings

    @classmethod
    def from_pretrained(cls, model, lokr_path, **kwargs):
        lokr_config = kwargs.pop("lokr_config", None)
        # init lokr config & lokr model
        if not isinstance(lokr_config, LoKrConfig):
            lokr_config = LoKrConfig.from_pretrained(lokr_path)
        # define a new variable to conserve original lora_config.tensor_parallel_degree value which will update while initializing lora model
        lokr_config_tensor_parallel_degree = lokr_config.tensor_parallel_degree
        lokr_model = cls(model, lokr_config)

        lokr_model_index_file = os.path.join(lokr_path, SAFE_PEFT_WEIGHTS_INDEX_NAME)
        if os.path.exists(lokr_model_index_file):
            # load safetensors format file.
            resolved_archieve_file, sharded_metadata = get_checkpoint_shard_files(
                pretrained_model_name_or_path=lokr_path,
                index_filename=lokr_model_index_file,
            )
            loaded_keys = sharded_metadata["all_checkpoint_keys"]
            expected_keys = set(lokr_model.get_trainable_state_dict().keys())

            missing_keys = expected_keys - set(loaded_keys)
            if len(missing_keys) > 0:
                raise ValueError(f"missing_keys: {missing_keys}")

            error_msgs = []
            for shard_file in resolved_archieve_file:
                pre_tensor_parallel_split = False
                if model.config.tensor_parallel_degree > 1:
                    pre_tensor_parallel_split = True
                    tp_actions = lokr_model._get_tensor_parallel_convert_actions(loaded_keys, is_split=True)
                state_dict = load_state_dict(
                    shard_file, tp_actions if pre_tensor_parallel_split else None, expected_keys
                )
                error_msgs += _load_state_dict_into_model(lokr_model.model, state_dict, "")
                del state_dict
                gc.collect()

            if len(error_msgs) > 0:
                error_msg = "\n\t".join(error_msgs)
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {lokr_model.__class__.__name__}:\n\t{error_msg}"
                )

            return lokr_model

        # define lokr weight name
        if lokr_config_tensor_parallel_degree > 1:
            lokr_weight_name = _add_variant(LOKR_WEIGHTS_NAME, f"tp{model.config.tensor_parallel_rank:0>2d}")
        else:
            lokr_weight_name = LOKR_WEIGHTS_NAME

        # load and set lokr weight parameter
        lokr_weight_path = os.path.join(lokr_path, lokr_weight_name)
        if os.path.exists(lokr_weight_path):
            # load lokr weight parameter
            lokr_state_dict = paddle.load(lokr_weight_path, return_numpy=True)
            logger.info(f"Loading the LoKR weights from {lokr_weight_path}")

            if (
                lokr_config_tensor_parallel_degree > 1
                and lokr_config_tensor_parallel_degree != model.config.tensor_parallel_degree
            ):
                raise NotImplementedError(
                    f"{lokr_config_tensor_parallel_degree} is not equal to {model.config.tensor_parallel_degree}. Please merge LoKR weights first."
                )

            # convert parameters to tensor parallel for mp model
            if lokr_config_tensor_parallel_degree <= 1 and model.config.tensor_parallel_degree > 1:
                lokr_state_dict = lokr_model._convert_tensor_parallel(lora_state_dict=lokr_state_dict)

            # set lora state dict
            lokr_model.set_state_dict(lokr_state_dict)
        else:
            logger.error(f"LoKR weights not found under {lokr_path}, creating LoKR weights from scratch")

        return lokr_model

    def set_state_dict(self, state_dict):
        import warnings

        warnings.filterwarnings(
            action="ignore", message=".*Skip loading for.*", category=Warning, lineno=0, append=False
        )
        self.model.set_state_dict(state_dict)
        logger.info("Load lokr weight successfully")

    def _merge_trainable_tensor_parallel(self, trainable_state_dict):
        trainable_name_action_mappings = self._get_tensor_parallel_convert_actions(
            trainable_state_dict.keys(), is_split=False
        )

        hcg = paddle.distributed.fleet.get_hybrid_communicate_group()
        mp_group = hcg.get_model_parallel_group()
        is_dst = paddle.distributed.get_rank(mp_group) == 0

        for key in trainable_state_dict:
            tensor = trainable_state_dict[key]
            if key in trainable_name_action_mappings:
                if get_env_device() == "xpu":
                    ret = distributed_allgather(tensor, group=mp_group, offload=True)
                else:
                    ret = distributed_gather(tensor, group=mp_group, offload=True)
                action = trainable_name_action_mappings[key]
                if key in self.lora_split_mapping and not self.lora_split_mapping[key] and "_scale" in key and is_dst:
                    ret = paddle.to_tensor(ret)
                    tensor = paddle.max(ret, axis=0)
                else:
                    tensor = action(ret) if is_dst else None
                trainable_state_dict[key] = tensor
            else:
                trainable_state_dict[key] = tensor.cpu().numpy() if is_dst else None

        return trainable_state_dict

    def _get_tensor_parallel_convert_actions(self, loaded_keys, is_split=True, ignore_error=False, config=None):
        if config is None:
            config = self.model.config
        specific_name_action_mappings = self._get_tensor_parallel_mappings(config, is_split=is_split)
        name_action_mappings = self.model._get_tensor_parallel_mappings(config, is_split=is_split)
        state_keys_map = ConversionMixin._resolve_prefix_keys(
            name_action_mappings.keys(), self.model.state_dict().keys(), ignore_error=ignore_error
        )
        for k, v in state_keys_map.items():
            if v in loaded_keys:
                specific_name_action_mappings[v] = name_action_mappings[k]
        return specific_name_action_mappings

    def _convert_tensor_parallel(self, lokr_state_dict):
        lokr_name_action_mappings = self._get_tensor_parallel_convert_actions(lokr_state_dict.keys(), is_split=True)

        for name, action in lokr_name_action_mappings.items():
            if name in lokr_state_dict:
                tensor = lokr_state_dict.pop(name)
                lokr_state_dict[name] = action(tensor)
            else:
                logger.warning(f"{name} not found in lokr_state_dict!")
        return lokr_state_dict

    def save_pretrained(self, save_directory: str, merge_tensor_parallel: bool = False, **kwargs):
        save_model_config = kwargs.get("save_model_config", True)

        if self.is_pipelinemodel:
            self.model._single_to_pp_mapping = None
        if self.quantized and merge_tensor_parallel and self.lokr_config.tensor_parallel_degree > 1:
            merge_tensor_parallel = False
            logger.warning(
                "Quantized strategy does not support merge_tensor_parallel. Set merge_tensor_parallel to False."
            )
        if self.is_pipelinemodel and merge_tensor_parallel and self.lokr_config.tensor_parallel_degree > 1:
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

        lokr_config_to_save = LoKrConfig(**self.lokr_config.to_dict())

        if merge_tensor_parallel and lokr_config_to_save.tensor_parallel_degree > 1:
            trainable_state_dict = self.get_trainable_state_dict()
            trainable_state_dict = self._merge_trainable_tensor_parallel(trainable_state_dict)
            if not is_main_process:
                logger.info("Saving with merge_tensor_parallel, tensor_parallel_rank > 0 don't need save")
                return
            if variant is not None and "tp" in variant:
                variant = "_".join([x for x in variant.split("_") if "tp" not in x])
            lokr_config_to_save.tensor_parallel_degree = -1
        else:
            trainable_state_dict = self.get_trainable_state_dict()
            if lokr_config_to_save.tensor_parallel_degree > 1:
                if variant is None:
                    variant = weight_name_suffix()

        # save lokr weight
        lokr_weight_name = _add_variant(LOKR_WEIGHTS_NAME, variant)
        weight_filename = os.path.join(save_directory, lokr_weight_name)
        paddle.save(trainable_state_dict, weight_filename)

        # save lokr config
        if is_main_process:
            lokr_config_to_save.save_pretrained(save_directory)
            if save_model_config:
                model_config_to_save = copy.deepcopy(self.model.config)
                if merge_tensor_parallel:
                    model_config_to_save.tensor_parallel_degree = -1
                model_config_to_save.save_pretrained(save_directory)

    def _find_and_replace_module(self, model, module_name, lokr_config, enable_lokr):
        parent_module = model
        attribute_chain = module_name.split(".")
        for name in attribute_chain[:-1]:
            parent_module = getattr(parent_module, name)
        module = getattr(parent_module, attribute_chain[-1])
        lokr_module = None
        if isinstance(module, nn.Linear):
            lokr_module = LoKrLinear(
                in_features=module.weight.shape[0],
                out_features=module.weight.shape[1],
                lora_dim=lokr_config.lora_dim,
                decompose_both=lokr_config.decompose_both,
                lokr_alpha=lokr_config.lokr_alpha,
                factor=lokr_config.factor,
                bias_attr=False if module.bias is None else None,
            )
            self.quantized = True
        if lokr_module is None:
            raise ValueError("LoKr strategy only supports paddle.nn.Linear right now")
        if getattr(lokr_module, "quant_weight", None) is not None:
            lokr_module.quant_weight = module.quant_weight
            if getattr(lokr_module, "quant_scale", None) is not None:
                lokr_module.quant_scale = module.quant_scale
            if getattr(lokr_module, "qquant_scale", None) is not None:
                lokr_module.qquant_scale = module.qquant_scale
            if getattr(lokr_module, "double_quant_scale", None) is not None:
                lokr_module.double_quant_scale = module.double_quant_scale
            if getattr(lokr_module, "quant_sacle_offset", None) is not None:
                lokr_module.quant_sacle_offset = module.quant_sacle_offset
        else:
            lokr_module.weight = module.weight
        if module.bias is not None:
            lokr_module.bias = module.bias
        setattr(parent_module, attribute_chain[-1], lokr_module)

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
            # get lokr parameter & QAT scale parameter
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

    def mark_only_lokr_as_trainable(self) -> None:
        for _, layer in self.model.named_sublayers():
            if (
                isinstance(layer, LoKrLinear)
                # or isinstance(layer, LoRAConv2D)
                # or isinstance(layer, ColumnParallelLoRALinear)
                # or isinstance(layer, RowParallelLoRALinear)
                # or isinstance(layer, ColumnSequenceParallelLoRALinear)
                # or isinstance(layer, RowSequenceParallelLoRALinear)
                # or (QuantizationLoRALinear is not None and isinstance(layer, QuantizationLoRALinear))
                # or (
                #     ColumnParallelQuantizationLoRALinear is not None
                #     and isinstance(layer, ColumnParallelQuantizationLoRALinear)
                # )
                # or (
                #     RowParallelQuantizationLoRALinear is not None
                #     and isinstance(layer, RowParallelQuantizationLoRALinear)
                # )
            ):
                for name, weight in layer.state_dict().items():
                    if self.lokr_config.trainable_bias in ["lokr", "all"] and "bias" in name:
                        weight.stop_gradient = False
                    elif "lokr" in name:
                        weight.stop_gradient = False
                    else:
                        weight.stop_gradient = True
            else:
                for name, weight in layer.state_dict().items():
                    if self.lokr_config.trainable_bias == "all" and "bias" in name:
                        weight.stop_gradient = False
                    else:
                        weight.stop_gradient = True
        if self.lokr_config.trainable_modules is not None:
            for name, weight in self.model.state_dict().items():
                if any(
                    re.fullmatch(trainable_module, name) for trainable_module in self.lokr_config.trainable_modules
                ):
                    weight.stop_gradient = False

    def get_lokr_model(self, model: Union[PretrainedModel, nn.Layer], lokr_config: LoKrConfig):
        if lokr_config.target_modules is None:
            return model
        elif isinstance(lokr_config.target_modules, str):
            target_modules = [lokr_config.target_modules]
            if lokr_config.enable_lokr_list is None or (
                isinstance(lokr_config.enable_lokr_list, List)
                and all(isinstance(item, bool) for item in lokr_config.enable_lokr_list)
            ):
                enable_lokr_list = [lokr_config.enable_lokr_list]
            else:
                raise TypeError(
                    f"Invalid `enable_lokr_list` value: {lokr_config.enable_lokr_list}. Since `target_modules` is `str`, `enable_lokr_list` must be `None` or `List[bool]`"
                )
        else:
            target_modules = lokr_config.target_modules
            if lokr_config.enable_lokr_list is None:
                enable_lokr_list = [None for _ in range(len(target_modules))]
            elif isinstance(lokr_config.enable_lokr_list, List):
                enable_lokr_list = lokr_config.enable_lokr_list
                if len(enable_lokr_list) != len(target_modules):
                    raise TypeError(
                        f"Invalid lokr_config.enable_lokr_list value: {lokr_config.enable_lokr_list}. Since lokr_config.target_modules is `List[str]`, `enable_lokr_list` should have the same length as `target_modules`"
                    )
                for enable_lokr in enable_lokr_list:
                    if not (
                        enable_lokr is None
                        or (isinstance(enable_lokr, List) and all(isinstance(item, bool) for item in enable_lokr))
                    ):
                        raise TypeError(
                            f"Invalid `enable_lokr_list` value: {lokr_config.enable_lokr_list}. Since `target_modules` is `List[str]`, `enable_lork_list` must be `None` or  `List[Optional[List[bool]]]`"
                        )
            else:
                raise TypeError(
                    f"Invalid `enable_lokr_list` value: {lokr_config.enable_lokr_list}. Since `target_modules` is `List[str]`, `enable_lokr_list` must be `None` or `List[Optional[List[bool]]]`"
                )

        for target_module, enable_lokr in zip(target_modules, enable_lokr_list):
            for i in model.named_sublayers():
                module_name = i[0]
                if re.fullmatch(target_module, module_name):
                    self._find_and_replace_module(model, module_name, lokr_config, enable_lokr)
        return model

    def restore_original_model(self):
        # make sure W and lora weights are not merged before we restore the original model

        for layer_name, layer in self.model.named_sublayers():
            if isinstance(layer, LoKrLinear):
                self._find_and_restore_module(layer_name)
            # elif (
            #     isinstance(layer, ColumnParallelLoRALinear)
            #     or isinstance(layer, ColumnSequenceParallelLoRALinear)
            #     or isinstance(layer, LoRAConv2D)
            #     or isinstance(layer, RowParallelLoRALinear)
            #     or isinstance(layer, RowSequenceParallelLoRALinear)
            #     or (QuantizationLoRALinear is not None and isinstance(layer, QuantizationLoRALinear))
            #     or (
            #         ColumnParallelQuantizationLoRALinear is not None
            #         and isinstance(layer, ColumnParallelQuantizationLoRALinear)
            #     )
            #     or (
            #         RowParallelQuantizationLoRALinear is not None
            #         and isinstance(layer, RowParallelQuantizationLoRALinear)
            #     )
            # ):
            #     raise NotImplementedError(f"{layer} restoration is not supported yet.")
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

    def disable_lokr(self):
        for _, layer in self.model.named_sublayers():
            if any(isinstance(layer, lokr_layer) for lokr_layer in AVAILABLE_LAYERS):
                layer.disable_lokr = True

    def enable_lokr(self):
        for _, layer in self.model.named_sublayers():
            if any(isinstance(layer, lokr_layer) for lokr_layer in AVAILABLE_LAYERS):
                layer.disable_lokr = False

    def merge(self):
        for _, layer in self.model.named_sublayers():
            if any(isinstance(layer, lokr_layer) for lokr_layer in AVAILABLE_LAYERS):
                layer.merge()

    def unmerge(self):
        for _, layer in self.model.named_sublayers():
            if any(isinstance(layer, lokr_layer) for lokr_layer in AVAILABLE_LAYERS):
                layer.unmerge()

    def get_model_config(
        self,
    ):
        return self.model_config.to_dict()
