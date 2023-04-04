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

import json
import math
import os
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.meta_parallel import ColumnParallelLinear

from ..transformers.model_utils import PretrainedModel
from ..utils.env import LORA_CONFIG_NAME, LORA_WEIGHT_FILE_NAME
from ..utils.log import logger

__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "LoRAMergedLinear",
    "LoRAModel",
]


class LoRALinear(nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

        # Actual trainable parameters
        if r > 0:
            self.lora_A = self.create_parameter(
                shape=[in_features, r],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.KaimingUniform(
                    negative_slope=math.sqrt(5), nonlinearity="leaky_relu"
                ),
            )
            self.lora_B = self.create_parameter(
                shape=[r, out_features],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.Constant(value=0.0),
            )
            self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            self.weight.stop_gradient = True

    def train(self):
        super().train()
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
                self.weight.set_value(new_weight)
            self.merged = False

    def eval(self):
        super().eval()
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
                self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, input: paddle.Tensor):
        result = F.linear(x=input, weight=self.weight, bias=self.bias, name=self.name)
        if self.r > 0 and not self.merged:
            result += (self.lora_dropout(input) @ self.lora_A @ self.lora_B) * self.scaling
        return result

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


class ColumnParallelLoRALinear(ColumnParallelLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs
    ):
        ColumnParallelLinear.__init__(self, in_features, out_features, **kwargs)
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

        # compatible
        self.name = self._name

        # Actual trainable parameters
        if r > 0:
            self.lora_A = self.create_parameter(
                shape=[in_features, r],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.KaimingUniform(
                    negative_slope=math.sqrt(5), nonlinearity="leaky_relu"
                ),
            )
            self.lora_B = self.create_parameter(
                shape=[r, self.output_size_per_partition],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.Constant(value=0.0),
            )
            self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            self.weight.stop_gradient = True

    def train(self):
        super().train()
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
                self.weight.set_value(new_weight)
            self.merged = False

    def eval(self):
        super().eval()
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
                self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, input: paddle.Tensor):
        input_mp = mp_ops._c_identity(input, group=self.model_parallel_group)
        result_mp = F.linear(x=input_mp, weight=self.weight, bias=self.bias, name=self.name)

        if self.r > 0 and not self.merged:
            input_a = self.lora_dropout(input_mp) @ self.lora_A
            delta_mp = (input_a @ self.lora_B) * self.scaling
            result_mp += delta_mp

        if self.gather_output and self.is_mp:
            result = mp_ops._c_concat(result_mp, group=self.model_parallel_group)
        else:
            result = result_mp
        return result

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


class LoRAMergedLinear(nn.Linear):
    # LoRA implemented in a dense layer  with merged linear weights for q, k, v
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        enable_lora: List[bool] = [False],
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        assert (
            out_features % len(enable_lora) == 0
        ), f"The length of enable_lora must divide out_features: {out_features} % {len(enable_lora)} != 0"
        self.r = r
        self.lora_alpha = lora_alpha
        if isinstance(enable_lora, List) and all(isinstance(item, bool) for item in enable_lora):
            self.enable_lora = enable_lora
        else:
            raise TypeError("enable_lora must be a list of bools")

        self.out_features = out_features
        self.in_features = in_features

        # Optional dropout
        if lora_dropout > 0.0 and any:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = self.create_parameter(
                shape=[in_features, r * sum(enable_lora)],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.KaimingUniform(
                    negative_slope=math.sqrt(5), nonlinearity="leaky_relu"
                ),
            )
            self.lora_B = self.create_parameter(
                shape=[out_features // len(enable_lora) * sum(enable_lora), r],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.Constant(value=0.0),
            )
            self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            self.weight.stop_gradient = True

    def zero_pad(self, x):
        # if enable_lora is all true, then there is no need to zero pad
        if all(self.enable_lora):
            return x
        else:
            split_output = paddle.split(x, sum(self.enable_lora), axis=-1)
            for index in range(len(self.enable_lora)):
                if self.enable_lora[index] is False:
                    split_output.insert(index, paddle.zeros_like(split_output[0]))
            concat_output = paddle.concat(split_output, axis=-1)
            return concat_output

    def train(self):
        super().train()
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                delta_weight = (
                    F.conv1d(
                        self.lora_A.transpose([1, 0]).unsqueeze(0),
                        self.lora_B.unsqueeze(-1),
                        groups=sum(self.enable_lora),
                    )
                    .squeeze(0)
                    .transpose([1, 0])
                )
                new_weight = self.weight - self.zero_pad(delta_weight * self.scaling)
                self.weight.set_value(new_weight)
            self.merged = False

    def eval(self):
        super().eval()
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_lora):
                delta_weight = (
                    F.conv1d(
                        self.lora_A.transpose([1, 0]).unsqueeze(0),
                        self.lora_B.unsqueeze(-1),
                        groups=sum(self.enable_lora),
                    )
                    .squeeze(0)
                    .transpose([1, 0])
                )
                new_weight = self.weight + self.zero_pad(delta_weight * self.scaling)
                self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, input: paddle.Tensor):
        result = F.linear(x=input, weight=self.weight, bias=self.bias, name=self.name)
        if self.r > 0 and any(self.enable_lora) and not self.merged:
            input_a = self.lora_dropout(input) @ self.lora_A
            if len(input_a.shape) == 2:
                delta = (
                    F.conv1d(
                        input_a.transpose([1, 0]).unsqueeze(0), self.lora_B.unsqueeze(-1), groups=sum(self.enable_lora)
                    )
                    .squeeze(0)
                    .transpose([1, 0])
                )
            elif len(input_a.shape) == 3:
                delta = (
                    F.conv1d(input_a.transpose([0, 2, 1]), self.lora_B.unsqueeze(-1), groups=sum(self.enable_lora))
                ).transpose([0, 2, 1])
            else:
                raise NotImplementedError("LoRAMergedLinear only support 2D or 3D input features")

            result += self.zero_pad(delta * self.scaling)
        return result

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


class ColumnParallelLoRAMergedLinear(ColumnParallelLinear):
    # LoRA implemented in a dense layer  with merged linear weights for q, k, v
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        enable_lora: List[bool] = [False],
        **kwargs
    ):
        ColumnParallelLinear.__init__(self, in_features, out_features, **kwargs)
        assert (
            self.output_size_per_partition % len(enable_lora) == 0
        ), f"The length of enable_lora must divide out_features: {self.output_size_per_partition} % {len(enable_lora)} != 0"
        self.r = r
        self.lora_alpha = lora_alpha
        if isinstance(enable_lora, List) and all(isinstance(item, bool) for item in enable_lora):
            self.enable_lora = enable_lora
        else:
            raise TypeError("enable_lora must be a list of bools")

        self.out_features = out_features
        self.in_features = in_features

        # Optional dropout
        if lora_dropout > 0.0 and any:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

        # compatible
        self.name = self._name

        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = self.create_parameter(
                shape=[in_features, r * sum(enable_lora)],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.KaimingUniform(
                    negative_slope=math.sqrt(5), nonlinearity="leaky_relu"
                ),
            )
            self.lora_B = self.create_parameter(
                shape=[self.output_size_per_partition // len(enable_lora) * sum(enable_lora), r],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.Constant(value=0.0),
            )
            self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            self.weight.stop_gradient = True

    def zero_pad(self, x):
        # if enable_lora is all true, then there is no need to zero pad
        if all(self.enable_lora):
            return x
        else:
            split_output = paddle.split(x, sum(self.enable_lora), axis=-1)
            for index in range(len(self.enable_lora)):
                if self.enable_lora[index] is False:
                    split_output.insert(index, paddle.zeros_like(split_output[0]))
            concat_output = paddle.concat(split_output, axis=-1)
            return concat_output

    def train(self):
        super().train()
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                delta_weight = (
                    F.conv1d(
                        self.lora_A.transpose([1, 0]).unsqueeze(0),
                        self.lora_B.unsqueeze(-1),
                        groups=sum(self.enable_lora),
                    )
                    .squeeze(0)
                    .transpose([1, 0])
                )
                new_weight = self.weight - self.zero_pad(delta_weight * self.scaling)
                self.weight.set_value(new_weight)
            self.merged = False

    def eval(self):
        super().eval()
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_lora):
                delta_weight = (
                    F.conv1d(
                        self.lora_A.transpose([1, 0]).unsqueeze(0),
                        self.lora_B.unsqueeze(-1),
                        groups=sum(self.enable_lora),
                    )
                    .squeeze(0)
                    .transpose([1, 0])
                )
                new_weight = self.weight + self.zero_pad(delta_weight * self.scaling)
                self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, input: paddle.Tensor):
        # [batch_size, *, in_features]
        input_mp = mp_ops._c_identity(input, group=self.model_parallel_group)
        # [batch_size, *, out_features_per_partition]
        result_mp = F.linear(x=input_mp, weight=self.weight, bias=self.bias, name=self.name)
        if self.r > 0 and any(self.enable_lora) and not self.merged:
            input_a = self.lora_dropout(input_mp) @ self.lora_A
            if len(input_a.shape) == 2:
                delta_mp = (
                    F.conv1d(
                        input_a.transpose([1, 0]).unsqueeze(0), self.lora_B.unsqueeze(-1), groups=sum(self.enable_lora)
                    )
                    .squeeze(0)
                    .transpose([1, 0])
                )
            elif len(input_a.shape) == 3:
                delta_mp = (
                    F.conv1d(input_a.transpose([0, 2, 1]), self.lora_B.unsqueeze(-1), groups=sum(self.enable_lora))
                ).transpose([0, 2, 1])
            else:
                raise NotImplementedError("LoRAMergedLinear only support 2D or 3D input features")
            # [batch_size, *, out_features_per_partition]
            result_mp += self.zero_pad(delta_mp * self.scaling)

        if self.gather_output and self.is_mp:
            result_mp_list = paddle.split(result_mp, len(self.enable_lora), axis=-1)
            result_list = []
            for result_mp in result_mp_list:
                result_list.append(mp_ops._c_concat(result_mp, group=self.model_parallel_group))
            # [batch_size, *, out_features]
            result = paddle.concat(result_list, axis=-1)
        else:
            result = result_mp

        return result

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


@dataclass
class LoRAConfig:
    """
    This is the configuration class to store the configuration of a [`LoRAModel`].
    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    trainable_bias: Optional[str] = field(
        default=None, metadata={"help": "Define trainable bias parameters for the Lora model."}
    )
    enable_lora_list: Optional[Union[List[bool], List[Optional[List[bool]]]]] = field(
        default=None,
        metadata={
            "help": "Provides fine-grained control over `MergedLoRALinear`. If None, `LoRALinear` is used instead."
        },
    )

    @property
    def __dict__(self):
        return asdict(self)

    def to_dict(self):
        return self.__dict__

    def save_pretrained(self, save_directory):
        r"""
        This method saves the configuration of your adapter model in a directory.
        Args:
            save_directory (`str`):
                The directory where the configuration will be saved.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        output_dict = self.__dict__
        output_path = os.path.join(save_directory, LORA_CONFIG_NAME)

        # save it
        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        This method loads the configuration of your adapter model from a directory.
        Args:
            pretrained_model_name_or_path (`str`):
                The directory or the hub-id where the configuration is saved.
            **kwargs:
                Additional keyword arguments passed along to the child class initialization.
        """
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, LORA_CONFIG_NAME)):
            config_file = os.path.join(pretrained_model_name_or_path, LORA_CONFIG_NAME)
        else:
            raise ValueError(f"Can't find lora_config.json at '{pretrained_model_name_or_path}'")

        loaded_attributes = cls.from_json_file(config_file)

        config = cls(**kwargs)

        for key, value in loaded_attributes.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    @classmethod
    def from_json_file(cls, path_json_file):
        r"""
        Loads a configuration file from a json file.
        Args:
            path_json_file (`str`):
                The path to the json file.
        """
        with open(path_json_file, "r") as file:
            json_object = json.load(file)

        return json_object


class LoRAModel(nn.Layer):
    def __init__(self, model, lora_config: LoRAConfig) -> None:
        super().__init__()
        self.lora_config = lora_config
        self.model = self.get_lora_model(model, lora_config)
        self.forward = self.model.forward

    @classmethod
    def from_pretrained(cls, model, lora_path):
        lora_config = LoRAConfig.from_pretrained(lora_path)
        lora_model = cls(model, lora_config)
        lora_weight_path = os.path.join(lora_path, LORA_WEIGHT_FILE_NAME)
        if os.path.exists(lora_weight_path):
            logger.info(f"Loading the LoRA weights from {lora_weight_path}")
            lora_state_dict = paddle.load(lora_weight_path)
            lora_model.model.set_state_dict(lora_state_dict)
        else:
            logger.info(f"LoRA weights not found under {lora_path}, creating LoRA weights from scratch")
        return lora_model

    def save_pretrained(self, save_directory: str, **kwargs):
        assert not os.path.isfile(
            save_directory
        ), f"Saving directory ({save_directory}) should be a directory, not a file"
        os.makedirs(save_directory, exist_ok=True)

        self.lora_config.save_pretrained(save_directory)
        weight_filename = os.path.join(save_directory, LORA_WEIGHT_FILE_NAME)
        trainable_state_dict = self.get_trainable_state_dict()
        paddle.save(trainable_state_dict, weight_filename)

    def _find_and_replace_module(self, model, module_name, lora_config, enable_lora):
        parent_module = model
        attribute_chain = module_name.split(".")
        for name in attribute_chain[:-1]:
            parent_module = getattr(parent_module, name)
        module = getattr(parent_module, attribute_chain[-1])
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
                )
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
                )

        lora_module.weight = module.weight
        if module.bias is not None:
            lora_module.bias = module.bias
        setattr(parent_module, attribute_chain[-1], lora_module)

    def get_trainable_state_dict(self):
        trainable_state_dict = OrderedDict()
        for name, weight in self.model.state_dict().items():
            if not weight.stop_gradient:
                trainable_state_dict[name] = weight
        return trainable_state_dict

    def print_trainable_parameters(self) -> None:
        freeze_numel = 0
        trainable_numel = 0
        for _, weight in self.model.state_dict().items():
            if weight.stop_gradient:
                freeze_numel += weight.numel().numpy()[0]
            else:
                trainable_numel += weight.numel().numpy()[0]
        logger.info(
            f"Frozen parameters: {freeze_numel:.2e} || Trainable parameters:{trainable_numel:.2e} || Total parameters:{freeze_numel+trainable_numel:.2e}|| Trainable:{trainable_numel / (freeze_numel+trainable_numel):.2%}"
        )

    def mark_only_lora_as_trainable(self) -> None:
        for _, layer in self.model.named_sublayers():
            if (
                isinstance(layer, LoRALinear)
                or isinstance(layer, ColumnParallelLoRALinear)
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

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Layer's logic
        except AttributeError:
            return getattr(self.model, name)
