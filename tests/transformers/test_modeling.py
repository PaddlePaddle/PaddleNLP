# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
from paddlenlp.transformers import TinyBertModel


class TestModeling(unittest.TestCase):
    """Test PretrainedModel single time, not in Transformer models"""

    def test_from_pretrained_with_load_as_state_np_params(self):
        """init model with `load_state_as_np` params"""
        TinyBertModel.from_pretrained("tinybert-4l-312d", load_state_as_np=True)
