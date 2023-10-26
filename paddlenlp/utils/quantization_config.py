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
import json
from dataclasses import dataclass


@dataclass
class QuantizationConfig:
    def __init__(
        self,
        weight_quantize_algo=None,
        quant_type=None,
        shift=False,
        smooth=False,
        shift_smooth_all_linears=False,
        quant_round_type=0,
        llm_int8_threshold=6.0,
    ):
        if weight_quantize_algo is not None and weight_quantize_algo not in [
            "weight_only_int8",
            "weight_only_int4",
            "llm.int8",
            "a8w8",
        ]:
            raise ValueError(
                f"weight_quantize_algo:{weight_quantize_algo} not in supported list ['weight_only_int8', 'weight_only_int4', 'llm.int8', 'a8w8']"
            )
        if quant_type is not None and quant_type not in ["weight_only_int8", "weight_only_int4", "a8w8"]:
            raise ValueError(
                f"quant_type:{quant_type} not in supported list ['weight_only_int8', 'weight_only_int4', 'a8w8']"
            )
        self.weight_quantize_algo = weight_quantize_algo
        self.quant_type = quant_type
        self.shift = shift
        self.smooth = smooth
        self.shift = shift
        self.shift_smooth_all_linears = shift_smooth_all_linears
        self.quant_round_type = quant_round_type
        self.llm_int8_threshold = llm_int8_threshold

    def is_weight_quantize(self):
        if self.weight_quantize_algo in ["weight_only_int8", "weight_only_int4", "llm.int8", "a8w8"]:
            return True
        else:
            return False

    def is_support_merge_tensor_parallel(self):
        if self.weight_quantize_algo in ["weight_only_int8", "weight_only_int4", "llm.int8", "a8w8"]:
            return False
        else:
            return True

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        """
        Instantiates QuantizationConfig from dict
        """
        config = cls(**config_dict)

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    def to_json_file(self, json_file_path):
        """
        Save this instance to a JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self, use_diff=True):
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_diff_dict(self):
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = QuantizationConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict
