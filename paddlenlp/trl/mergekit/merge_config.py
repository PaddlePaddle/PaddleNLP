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

import json
import os
from dataclasses import asdict, dataclass, field

from paddlenlp.utils.log import logger

from ...utils.env import MERGE_CONFIG_NAME


@dataclass
class MergeConfig:
    """
    This is the configuration class to store the configuration of a [`LoRAModel`].
    Args:
        linear_ratio (`float`):
    """

    merge_type: str = field(default="linear", metadata={"help": "The type of merge strategy."})
    linear_ratio: float = field(default=0.5, metadata={"help": "Linear merge ratio."})
    merge_preifx: str = field(default="model", metadata={"help": "Prefix name: model or master_weights"})
    device: str = field(default="cpu", metadata={"help": "Device to use for the merge.ex cpu、 gpu、low_gpu_mem"})
    n_process: int = field(default=1, metadata={"help": "Number of processes to use for the merge."})
    dtype: str = field(default="float32", metadata={"help": "Data type to use for the merge."})
    dot_threshold: float = field(
        default=0.99, metadata={"help": "Threshold for considering the two vectors as colinear."}
    )

    def __post_init__(self):
        if self.device != "cpu":
            logger.warning(f"Currently only support cpu device, but got {self.device}. Setting `device` to `cpu`.")
            self.device = "cpu"
        if self.merge_preifx == "master_weights" and self.dtype != "float32":
            logger.warning(
                f"Currently only support float32 data type for master weights, but got {self.dtype}. Setting `dtype` to `float32`."
            )
            self.dtype = "float32"

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
        output_path = os.path.join(save_directory, MERGE_CONFIG_NAME)

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
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, MERGE_CONFIG_NAME)):
            config_file = os.path.join(pretrained_model_name_or_path, MERGE_CONFIG_NAME)
        else:
            raise ValueError(f"Can't find lora_config.json at '{pretrained_model_name_or_path}'")

        loaded_attributes = cls.from_json_file(config_file)
        loaded_attributes.pop("scaling", None)

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
