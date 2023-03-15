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
import os
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..utils.env import LORA_CONFIG_NAME
from ..utils.log import logger

__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "get_lora_model",
    "mark_only_lora_as_trainable",
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
            )
            self.lora_B = self.create_parameter(
                shape=[r, out_features],
                dtype=self._dtype,
                is_bias=False,
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.stop_gradient = True
            self.bias.stop_gradient = True

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
        if self.r > 0 and not self.merged:
            result = F.linear(x=input, weight=self.weight, bias=self.bias, name=self.name)
            if self.r > 0:
                result += (self.lora_dropout(input) @ self.lora_A @ self.lora_B) * self.scaling
            return result
        else:
            return F.linear(x=input, weight=self.weight, bias=self.bias, name=self.name)

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


# TODO (this is tmp API. will formalize before release)
def _find_and_replace_module(model, module_name, lora_config):
    parent_module = model
    attribute_chain = module_name.split(".")
    for name in attribute_chain[:-1]:
        parent_module = getattr(parent_module, name)
    module = getattr(parent_module, attribute_chain[-1])
    lora_module = LoRALinear(
        in_features=module.weight.shape[0],
        out_features=module.weight.shape[1],
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        merge_weights=lora_config.merge_weights,
    )
    setattr(parent_module, attribute_chain[-1], lora_module)


def mark_only_lora_as_trainable(model: nn.Layer) -> None:
    freeze_numel, trainable_numel = 0, 0
    for name, weight in model.state_dict().items():
        if "lora" not in name:
            weight.stop_gradient = True
            freeze_numel += weight.numel().numpy()[0]
        else:
            trainable_numel += weight.numel().numpy()[0]
    logger.info(f"{freeze_numel:.2e} parameters are frozen, {trainable_numel:.2e} LoRA parameters are trainable")


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
            raise ValueError(f"Can't find config.json at '{pretrained_model_name_or_path}'")

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


# TODO (this is tmp API. will formalize before release)
def get_lora_model(model, lora_config: LoRAConfig):
    target_modules = lora_config.target_modules
    for target_module in target_modules:
        for i in model.named_sublayers():
            module_name = i[0]
            if re.fullmatch(target_module, module_name):
                _find_and_replace_module(model, module_name, lora_config)
    return model
