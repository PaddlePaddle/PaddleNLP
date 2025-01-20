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

from paddlenlp.peft.utils import PeftType

from ....utils.env import VERA_CONFIG_NAME
from ...config import PeftConfig


@dataclass
class VeRAConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`VeRAModel`].
    Args:
        r (`int`): vera attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply vera to.
        trainable_modules (`List[str]`): The names of the modules to train when applying vera.
        vera_alpha (`float`): The alpha parameter for vera scaling.
        vera_dropout (`float`): The dropout probability for vera layers.
        bias (`str`):
            Bias type for Vera. Can be 'none', 'all' or 'vera_only'. If 'all' or 'vera_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
    """

    r: int = field(default=8, metadata={"help": "vera attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with vera."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Vera. Can be 'none', 'all' or 'vera_only'"})

    trainable_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to train when applying with vera."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    projection_prng_key: int = field(
        default=0,
        metadata={
            "help": (
                "Vera PRNG init key. Used for initialising vera_A and vera_B for new models or when loading a "
                "checkpoint that did not include these projections."
            )
        },
    )
    save_projection: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to save the vera_A / vera_B projections in the state dict alongside per layer lambda_b / "
                "lambda_d weights. This will increase the size of the checkpoint, but guarantee that we can reload "
                "the checkpoint on all system configurations."
            )
        },
    )
    vera_alpha: int = field(default=8, metadata={"help": "vera alpha"})
    vera_dropout: float = field(default=0.0, metadata={"help": "vera dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    trainable_bias: Optional[str] = field(
        default=None, metadata={"help": "Define trainable bias parameters for the vera model."}
    )
    d_initial: float = field(default=0.1, metadata={"help": "Initial init value for d vector."})
    tensor_parallel_degree: int = field(default=-1, metadata={"help": "1 for not use tensor parallel"})
    dtype: Optional[str] = field(default=None, metadata={"help": "The data type of tensor"})
    head_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "The model multi head dimension.Only for veraMergedLinear and ColumnParallelveraMergedLinear."
        },
    )
    do_qat: bool = field(default=False, metadata={"help": "Whether the vera model would do quant-aware training"})
    base_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the base model to use."}
    )

    pissa_init: bool = field(default=False, metadata={"help": "Whether the vera weight initialized by pissa"})

    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the Vera layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.VERA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # # check for layers_to_transform and layers_pattern
        # if self.layers_pattern and not self.layers_to_transform:
        #     raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")
        if not self.save_projection:
            warnings.warn(
                "Specified to not save vera_A and vera_B within the state dictionary, instead they will be restored "
                "using the PRNG key store in `config.projection_prng_key`. Consider setting `config.save_projection` "
                "to `True` to guarantee restoring the checkpoint correctly on all system configurations."
            )

    @property
    def __dict__(self):
        return asdict(self)

    def to_dict(self):
        return self.__dict__

    # def save_pretrained(self, save_directory):
    #     r"""
    #     This method saves the configuration of your adapter model in a directory.
    #     Args:
    #         save_directory (`str`):
    #             The directory where the configuration will be saved.
    #     """
    #     if os.path.isfile(save_directory):
    #         raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

    #     os.makedirs(save_directory, exist_ok=True)

    #     output_dict = self.__dict__
    #     output_path = os.path.join(save_directory, VERA_CONFIG_NAME)

    #     # save it
    #     with open(output_path, "w") as writer:
    #         writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    #     r"""
    #     This method loads the configuration of your adapter model from a directory.
    #     Args:
    #         pretrained_model_name_or_path (`str`):
    #             The directory or the hub-id where the configuration is saved.
    #         **kwargs:
    #             Additional keyword arguments passed along to the child class initialization.
    #     """
    #     if os.path.isfile(os.path.join(pretrained_model_name_or_path, VERA_CONFIG_NAME)):
    #         config_file = os.path.join(pretrained_model_name_or_path, VERA_CONFIG_NAME)
    #     else:
    #         raise ValueError(f"Can't find vera_config.json at '{pretrained_model_name_or_path}'")

    #     loaded_attributes = cls.from_json_file(config_file)

    #     config = cls(**kwargs)

    #     for key, value in loaded_attributes.items():
    #         if hasattr(config, key):
    #             setattr(config, key, value)

    #     return config

    # @classmethod
    # def from_json_file(cls, path_json_file):
    #     r"""
    #     Loads a configuration file from a json file.
    #     Args:
    #         path_json_file (`str`):
    #             The path to the json file.
    #     """
    #     with open(path_json_file, "r") as file:
    #         json_object = json.load(file)

    #     return json_object
