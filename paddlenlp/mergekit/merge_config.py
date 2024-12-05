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
from typing import List, Optional

import paddle

from paddlenlp.utils.env import MERGE_CONFIG_NAME
from paddlenlp.utils.log import logger


@dataclass
class MergeConfig:
    """
    This is the configuration class to store the configuration of a [`MergeKit`].
    """

    # Common parameters
    device: str = field(default="cpu", metadata={"help": "Device to use for the merge.ex cpu、 gpu、low_gpu_mem"})
    tensor_type: str = field(
        default="np", metadata={"help": "Tensor type to use for the merge. Choose np(CPU Only) or pd (CPU/GPU)"}
    )
    n_process: int = field(default=1, metadata={"help": "Number of processes to use for the merge."})
    merge_preifx: str = field(default="model", metadata={"help": "Prefix name: model or master_weights"})
    merge_method: str = field(default=None, metadata={"help": "The merge strategy."})
    merge_type: str = field(default=None, metadata={"help": "The type of merge process."})
    sparsify_type: str = field(default=None, metadata={"help": "The type of sparsify process."})

    # Model
    model_name_or_path_list: Optional[List[str]] = field(
        default=None, metadata={"help": "Merge model name or path list"}
    )
    base_model_name_or_path: str = field(default=None, metadata={"help": "Base model name or path."})
    output_path: str = field(default=None, metadata={"help": "Base model name or path."})

    # merge parameters
    weight_list: Optional[List[float]] = field(
        default=None, metadata={"help": "Relative (or absolute if normalize=False) weighting of a given tensor"}
    )
    normalize: bool = field(default=False, metadata={"help": "Whether to normalize the weighting."})
    slerp_normalize_eps: float = field(default=1e-8, metadata={"help": "Slerp normalization epsilon value"})
    slerp_dot_threshold: float = field(
        default=0.9995,
        metadata={
            "help": "Slerp dot threshold. If dot value exceeds this threshold, then we consider them as colinear, so use linear instead."
        },
    )
    ties_elect_type: str = field(default="sum", metadata={"help": "The type of ties mask. 'sum' or 'count'"})

    # Sparsify parameters
    rescale: bool = field(default=True, metadata={"help": "Rescale the weights after sparsifying."})
    reserve_p: float = field(default=0.7, metadata={"help": "Random reserve probability for the sparsify model."})
    epsilon: float = field(default=0.14, metadata={"help": "Random reserve probability for the sparsify model."})
    drop_rate: float = field(default=0.7, metadata={"help": "Drop rate for the merge."})
    della_rate: float = field(default=0.2, metadata={"help": "Della rate for the merge."})

    def __post_init__(self):
        self.config_check()

    def config_check(self):
        if self.tensor_type not in ["pd", "np"]:
            raise ValueError(f"Unsupported tensor type: {self.tensor_type}. Please choose one from ['pd', 'np'].")
        if self.device != "cpu":
            logger.warning(f"Currently only support cpu device, but got {self.device}. Setting `device` to `cpu`.")
            self.device = "cpu"
            self.tensor_type = "np"

        if self.merge_method is None and self.merge_type is None:
            raise ValueError("Please specify the merge_method or merge_type")
        elif self.merge_method not in ["linear", "ties", "slerp", "della_linear", "della_ties", "dare", "dare_ties"]:
            raise ValueError(
                f"Unsupported merge strategy: {self.merge_method}. Please choose one from ['linear', 'slerp']."
            )
        if self.weight_list is None:
            self.weight_list = [1.0] * len(self.model_name_or_path_list)
            self.normalize = True
        if len(self.model_name_or_path_list) != len(self.weight_list):
            raise ValueError("The length of model_name_or_path_list and weight_list must be the same.")
        if self.output_path is None:
            raise ValueError("Please specify the output_path.")
        if self.reserve_p < 0 or self.reserve_p > 1:
            raise ValueError("reserve_p must be between 0 and 1.")
        if self.reserve_p <= self.epsilon / 2 or self.reserve_p >= (1 - self.epsilon):
            raise ValueError(
                f"Error: reserve_p +- epsilon/2 must be in the range (0, 1). reserve_p + epsilon/2 = {self.reserve_p + self.epsilon / 2 }, reserve_p - epsilon/2 = {self.reserve_p - self.epsilon / 2 }"
            )
        paddle.set_device(self.device)

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
