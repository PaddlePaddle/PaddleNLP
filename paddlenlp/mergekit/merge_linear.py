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
# import numpy as np
from .merge_sparsify import SparsificationMethod


class MergeLinear:
    def __init__(self, merge_config):
        self.merge_config = merge_config

    def merge_op(self, v_0, v_1):
        """
        Linear interpolation between two values.
        """
        if self.merge_config.sparsify_type is not None:
            sparsify = SparsificationMethod(self.merge_config)
            v_0 = sparsify.sparsify_method(v_0)
            v_1 = sparsify.sparsify_method(v_1)
            v_merge = (1 - self.merge_config.linear_ratio) * v_0 + self.merge_config.linear_ratio * v_1

        else:
            v_merge = (1 - self.merge_config.linear_ratio) * v_0 + self.merge_config.linear_ratio * v_1

        return v_merge

    def merge_dict(self, dict1, dict2):
        if dict1.keys() != dict2.keys():
            raise ValueError(f"The keys of two dictionaries are different. {dict1.keys()} != {dict2.keys()} ")
        dict_merge = {}
        for key in dict1:
            dict_merge[key] = self.merge_op(dict1[key], dict2[key])
        return dict_merge
