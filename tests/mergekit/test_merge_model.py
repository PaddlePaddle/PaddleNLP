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

# import copy
# import os
# import re
# import unittest
# from tempfile import TemporaryDirectory

# import numpy as np
# import paddle
# from parameterized import parameterized

# from paddlenlp.mergekit import MergeConfig, SparsifyMethod

# class TestSparsifyMethod(unittest.TestCase):
#     def setUpClass(cls):
#         cls.pd_param_model_name = "bert-base-uncased"
#         cls.safetensors_model_name_or_path_list = []
#         cls.base_model_name_or_path = "bert-base-uncased"

#     @parameterized.expand([("slerp",), ("dare_linear",), ("lora",)])
#     def test_safetensor_model(self, merge_method):
#         with TemporaryDirectory() as tempdir:
#             merge_config = MergeConfig(
#                 merge_method=merge_method,
#                 model_name_or_path_list= self.safetensors_model_name_or_path_list,
#                 base_model_name_or_path=self.base_model_name_or_path,
#                 output_path=tempdir)
#             mergekit = MergeModel(merge_config)
#             mergekit.merge()
