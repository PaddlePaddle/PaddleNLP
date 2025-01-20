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

# from .vera_layers import VeRALinear
from paddlenlp.peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
)

from ....transformers.model_utils import PretrainedModel, _add_variant, dtype_guard
from ....utils.env import VERA_WEIGHTS_NAME
from ....utils.log import logger
from .._buffer_dict import BufferDict
from ..tuners_utils import _maybe_include_all_linear_layers, place_to_str
from .vera_config import VeRAConfig
from .vera_layers import Linear, VeraLayer


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def _kaiming_init(
    tensor_or_shape: Union[paddle.Tensor, tuple[int, ...]],
    generator: paddle.base.core.Generator,
) -> paddle.Tensor:
    """
    Kaiming Uniform Initialisation adapted to accept a `paddle.Generator` object for PRNG.

    Args:
        tensor_or_shape (`Union[paddle.Tensor, tuple[int, ...]]`):
            Tensor to initialise, or shape of new tensor to create and then initialise.
        generator: (`paddle.Generator`):
            Generator object that manages the state of the PRNG algorithm in use.

    Returns:
        `paddle.Tensor`: The initialised tensor.
    """
    import math

    if isinstance(tensor_or_shape, tuple):
        tensor = paddle.empty(tensor_or_shape)
    else:
        tensor = tensor_or_shape
    fan = _calculate_correct_fan(tensor, "fan_in")
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std

    with paddle.no_grad():
        return tensor.uniform_(-bound, bound)  # , generator=generator)


class VeRAModel(BaseTuner):
    """
    Creates Vector-based Random Matrix Adaptation (Vera) model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`VeraConfig`]): The configuration of the Vera model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The Vera model.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import VeraConfig, get_peft_model

        >>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> config = VeraConfig(r=128)
        >>> model = get_peft_model(base_model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`VeraConfig`]): The configuration of the Vera model.
    """

    prefix: str = "vera_lambda_"

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    def __init__(self, model, config: VeRAConfig, adapter_name: str, low_cpu_mem_usage: bool = False) -> None:
        # super().__init__()
        vera_config = config
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
        self.quantized = False
        self.vera_config = vera_config
        # if self.vera_config.dtype is None:
        #     self.vera_config.dtype = paddle.get_default_dtype()
        # with dtype_guard(self.vera_config.dtype):
        #     self.model = self.get_vera_model(model, vera_config)

        # self.is_pipelinemodel = False
        # if issubclass(type(self.model), PipelineLayer):
        #     raise NotImplementedError("vera don't support pipeline parallel now")
        # if vera_config.tensor_parallel_degree > 1:
        #     raise NotImplementedError("vera don't support tensor parallel now")
        # self.forward = self.model.forward

    @classmethod
    def from_pretrained(cls, model, vera_path, **kwargs):
        vera_config = kwargs.pop("vera_config", None)
        # init vera config & vera model
        if not isinstance(vera_config, VeRAConfig):
            vera_config = VeRAConfig.from_pretrained(vera_path)
        # define a new variable to conserve original vera_config.tensor_parallel_degree value which will update while initializing vera model
        vera_config_tensor_parallel_degree = vera_config.tensor_parallel_degree
        vera_model = cls(model, vera_config)

        vera_weight_name = VERA_WEIGHTS_NAME

        # load and set vera weight parameter
        vera_weight_path = os.path.join(vera_path, vera_weight_name)
        logger.info(f"vera weight path is {vera_weight_path}")
        if os.path.exists(vera_weight_path):
            # load vera weight parameter
            logger.info("vera_weight_path existed, loading vera weight parameter")

            vera_state_dict = paddle.load(vera_weight_path, return_numpy=True)
            logger.info(f"Loading the VeRA weights from {vera_weight_path}")

            if (
                vera_config_tensor_parallel_degree > 1
                and vera_config_tensor_parallel_degree != model.config.tensor_parallel_degree
            ):
                raise NotImplementedError(
                    f"{vera_config_tensor_parallel_degree} is not equal to {model.config.tensor_parallel_degree}. Please merge VeRA weights first."
                )

            # set vera state dict
            vera_model.set_state_dict(vera_state_dict)
        else:
            logger.error(f"VeRA weights not found under {vera_path}, creating VeRA weights from scratch")

        return vera_model

    def set_state_dict(self, state_dict):
        import warnings

        warnings.filterwarnings(
            action="ignore", message=".*Skip loading for.*", category=Warning, lineno=0, append=False
        )
        self.model.set_state_dict(state_dict)
        logger.info("Load vera weight successfully")

    def save_pretrained(self, save_directory: str, merge_tensor_parallel: bool = False, **kwargs):

        logger.info("save vera pretrained")
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

        logger.info(f"vera config to save is {vera_config_to_save}")

        trainable_state_dict = self.get_trainable_state_dict()

        # save vera weight
        vera_weight_name = _add_variant(VERA_WEIGHTS_NAME, variant)
        weight_filename = os.path.join(save_directory, vera_weight_name)
        paddle.save(trainable_state_dict, weight_filename)

        # save vera config
        if is_main_process:
            vera_config_to_save.save_pretrained(save_directory)
            if save_model_config:
                model_config_to_save = copy.deepcopy(self.model.config)
                if merge_tensor_parallel:
                    model_config_to_save.tensor_parallel_degree = -1
                model_config_to_save.save_pretrained(save_directory)

    def _find_and_replace_module(self, model, module_name, vera_config, enable_vera):
        parent_module = model
        attribute_chain = module_name.split(".")
        for name in attribute_chain[:-1]:
            parent_module = getattr(parent_module, name)
        module = getattr(parent_module, attribute_chain[-1])
        vera_module = None
        if enable_vera is None:
            if isinstance(module, nn.Linear):
                vera_module = Linear(
                    # pass the base linear module
                    base_linear_module=module,
                    in_features=module.weight.shape[0],
                    out_features=module.weight.shape[1],
                    r=vera_config.r,
                    vera_alpha=vera_config.vera_alpha,
                    vera_dropout=vera_config.vera_dropout,
                    bias_attr=False if module.bias is None else None,
                    pissa_init=vera_config.pissa_init,
                )

        if vera_module is None:
            raise ValueError(
                f"VeRA strategy only supports paddle.nn.Linear or paddle.distributed.fleet.meta_parallel.ColumnParallelLinear. {module}({module_name}) is not supported。"
            )

        if module.bias is not None:
            vera_module.bias = module.bias

        setattr(parent_module, attribute_chain[-1], vera_module)

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
            # get vera parameter
            if not weight.stop_gradient:
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
            if isinstance(layer, Linear):
                for name, weight in layer.state_dict().items():
                    if self.vera_config.trainable_bias in ["vera", "all"] and "bias" in name:
                        weight.stop_gradient = False
                    elif "vera" in name:
                        # notfreezeB=True, vera_b, vera_d, vera_B is trainable
                        # notfreezeB=False, vera_b, vera_d is trainable
                        if "vera_b" in name or "vera_d" in name:
                            weight.stop_gradient = False
                        elif "vera_B" in name and notfreezeB:
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
            enable_vera_list = [None]
        else:
            target_modules = vera_config.target_modules
            enable_vera_list = [None for _ in range(len(target_modules))]

        for target_module, enable_vera in zip(target_modules, enable_vera_list):
            for i in model.named_sublayers():
                module_name = i[0]
                if re.fullmatch(target_module, module_name):
                    self._find_and_replace_module(model, module_name, vera_config, enable_vera)
        return model

    def restore_original_model(self):
        for layer_name, layer in self.model.named_sublayers():
            if isinstance(layer, Linear):
                self._find_and_restore_module(layer_name)
            else:
                raise NotImplementedError(f"{layer} restoration is not supported yet.")
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

    # def _set_adapter_layers(self, enabled=True):
    #     for module in self.model.modules():
    #         if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):

    def _find_dim(self, config) -> tuple[int, int]:
        """
        Finds the largest input and output dimensions across linear layers that have been wrapped with VeRA.

        This will be used for determining the size of the shared vera_A and vera_B matrices.
        """
        model_config = self.get_model_config(self.model)

        peft_config = self._prepare_adapter_config(config, model_config)
        peft_config = _maybe_include_all_linear_layers(peft_config, self.model)

        largest_shape = None
        for key, module in self.model.named_sublayers():
            if not self._check_target_module_exists(peft_config, key):
                continue

            if isinstance(module, nn.Linear):
                # module_shape = module.out_features, module.in_features
                in_features, out_features = module.weight.shape
                module_shape = (out_features, in_features)
            elif isinstance(module, Conv1D):
                module_shape = module.weight.ds_shape if hasattr(module.weight, "ds_shape") else module.weight.shape
                module_shape = module_shape[::-1]
            else:
                continue

            if largest_shape is None:
                largest_shape = module_shape
                continue

            if module_shape != largest_shape:
                largest_shape = tuple(max(a, b) for a, b in zip(largest_shape, module_shape))

        if largest_shape is None:
            msg = "No layers types compatible with VeRA were found. Please check `peft_config.target_modules`."
            raise ValueError(msg)

        return largest_shape

    def _init_vera_A_vera_B(self, config: VeRAConfig, adapter_name: str) -> None:
        linear_out_dim, linear_in_dim = self._find_dim(config)

        # use of persistent to exclude vera_A and vera_B from the state dict if we choose not to save them.
        self.vera_A = BufferDict({}, persistent=config.save_projection)
        self.vera_B = BufferDict({}, persistent=config.save_projection)

        # deterministic init of vera_A and vera_B if we know the key
        # generator = paddle.Generator(device="cpu").manual_seed(config.projection_prng_key)
        generator = paddle.base.core.default_cpu_generator().manual_seed(config.projection_prng_key)
        vera_A = _kaiming_init((config.r, linear_in_dim), generator=generator)
        vera_B = _kaiming_init((linear_out_dim, config.r), generator=generator)

        self.vera_A[adapter_name] = vera_A
        self.vera_B[adapter_name] = vera_B

    def _pre_injection_hook(self, model: nn.Layer, config: VeRAConfig, adapter_name: str) -> None:
        self._init_vera_A_vera_B(config, adapter_name)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    @staticmethod
    def _check_target_module_exists(vera_config, key):
        return check_target_module_exists(vera_config, key)

    def _create_and_replace(
        self,
        vera_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        r = vera_config.r
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": r,
            "vera_dropout": vera_config.vera_dropout,
            "fan_in_fan_out": vera_config.fan_in_fan_out,
            "init_weights": vera_config.init_weights,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }
        kwargs["bias"] = bias

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name,
                self.vera_A,
                self.vera_B,
                r,
                vera_config.vera_dropout,
                vera_config.init_weights,
                d_initial=vera_config.d_initial,
            )
        else:
            new_module = self._create_new_module(vera_config, self.vera_A, self.vera_B, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                # new_module.requires_grad_(False)
                new_module.stop_gradients = True
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(place_to_str(child.weight.device))

        # meta = torch.device("meta")
        meta = None
        # dispatch to correct device
        for name, module in new_module.named_sublayers():
            if "vera_" in name:
                if not any(p.place == meta for p in module.parameters()):
                    module.to(place_to_str(child.weight.place))

    def _mark_only_adapters_as_trainable(self, model: nn.Layer) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            print(self.peft_config[active_adapter])
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "vera_only":
                for m in model.modules():
                    if isinstance(m, VeraLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    def disable_adapter_layers(self):
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    @staticmethod
    def _create_new_module(vera_config, vera_A, vera_B, adapter_name, target, **kwargs):
        # # avoid eager bnb import
        # if is_bnb_available():
        #     import bitsandbytes as bnb

        #     from .bnb import Linear8bitLt

        # if is_bnb_4bit_available():
        #     from .bnb import Linear4bit

        bias = kwargs.pop("bias", False)
        loaded_in_8bit = kwargs.get("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.get("loaded_in_4bit", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target_base_layer.state.has_fp16_weights,
                    "threshold": target_base_layer.state.threshold,
                    "index": target_base_layer.index,
                }
            )
            return Linear8bitLt(target, adapter_name, vera_A, vera_B, **eightbit_kwargs)
        elif loaded_in_4bit and isinstance(target_base_layer, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target_base_layer.compute_dtype,
                    "compress_statistics": target_base_layer.weight.compress_statistics,
                    "quant_type": target_base_layer.weight.quant_type,
                }
            )
            return Linear4bit(target, adapter_name, vera_A, vera_B, **fourbit_kwargs)
        elif isinstance(target_base_layer, paddle.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `paddle.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = vera_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = vera_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`paddle.nn.Linear`, `transformers.pypaddle_utils.Conv1D`."
            )
        # print(kwargs)
        # print(bias)
        # print(vera_A)
        # print(vera_B)

        new_module = Linear(
            target,
            vera_A,
            vera_B,
            adapter_name,
            bias_attr=bias,
            d_initial=vera_config.d_initial,
            **kwargs,
        )

        return new_module

    def set_adapter(self, adapter_name):
        import warnings

        for module in self.model.sublayers():
            if isinstance(module, VeraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name
