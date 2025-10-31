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
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Union

from ...utils.env import DISLORA_CONFIG_NAME


@dataclass
class DisLoRAConfig:
    """
    This is the configuration class to store the configuration of a [`DisLoRAModel`].
    Args:
        target_modules (`Union[List[str],str]`): The names of the modules to apply DisLoRA to.
        trainable_modules (`List[str]`): The names of the modules to train when applying DisLoRA.
        dislora_alpha (`float`): The alpha parameter for DisLoRA scaling.
        merge_weights (`bool`):
            Whether to merge the weights of the DisLoRA layers with the base transfoisrmer model in `eval` mode.
    """

    base_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the base model to use."}
    )
    r: int = field(default=8, metadata={"help": "DisLoRA attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with DisLoRA."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    trainable_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to train when applying with DisLoRA."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    dislora_alpha: int = field(default=12, metadata={"help": "DisLoRA alpha"})
    dislora_dropout: float = field(default=0.0, metadata={"help": "DisLoRA dropout"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the DisLoRA model"}
    )
    trainable_bias: Optional[str] = field(
        default=None, metadata={"help": "Define trainable bias parameters for the DisLoRA model."}
    )

    tensor_parallel_degree: int = field(default=-1, metadata={"help": "1 for not use tensor parallel"})
    dtype: Optional[str] = field(default=None, metadata={"help": "The data type of tensor"})

    dash_flag: int = field(  # characteristic
        default=50,
        metadata={"help": "The number of preheating steps before introducing additional low-rank updates"},
    )

    s_tsd: int = field(  # characteristic
        default=8,
        metadata={"help": "The number of top-k singular vectors dynamically selected after preheating"},
    )

    ortho_lambda: float = field(  # characteristic
        default=1,
        metadata={"help": "The weight of orthogonal regularization loss"},
    )
    prefer_small_sigma: bool = field(
        default=True,
        metadata={"help": "Whether to prioritize the smallest singular value in the top-k selection process"},
    )

    def __post_init__(self):

        if self.target_modules is None:
            raise ValueError("The target_modules must be specified as a string or a list of strings.")
        if self.r <= 0:
            raise ValueError("The rank r of LoRA must be greater than 0.")
        if self.dislora_alpha <= 0:
            raise ValueError("dislora_alpha must be greater than 0")
        if self.r < self.s_tsd:
            raise ValueError("The rank r of LoRA must be larger than the number of top-k singular values.")

    @property
    def scaling(self):
        return self.dislora_alpha / self.r

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
        output_dict["scaling"] = self.scaling
        output_path = os.path.join(save_directory, DISLORA_CONFIG_NAME)

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
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, DISLORA_CONFIG_NAME)):
            config_file = os.path.join(pretrained_model_name_or_path, DISLORA_CONFIG_NAME)
        else:
            raise ValueError(f"Can't find dislora_config.json at '{pretrained_model_name_or_path}'")

        loaded_attributes = cls.from_json_file(config_file)
        loaded_attributes.pop("scaling", None)

        merged_kwargs = {**loaded_attributes, **kwargs}
        config = cls(**merged_kwargs)

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
