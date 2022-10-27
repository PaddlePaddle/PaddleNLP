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

from unittest import TestCase
from paddlenlp.utils.converter import Converter


class TestConverter(TestCase):
    """test Converter base method"""

    def test_get_num_layer(self):
        """test `get_num_layer` method"""
        layers = [
            "embeddings_project.weight",
            "embeddings_project.bias",
        ]
        num_layer = Converter.get_num_layer(layers)
        self.assertIsNone(num_layer)

        layers = [
            "embeddings_project.weight",
            "embeddings_project.bias",
            "encoder.layer.0.attention.self.query.weight",
            "encoder.layer.1.attention.self.query.weight",
            "encoder.layer.6.attention.self.query.bias",
            "encoder.layer.11.attention.output.LayerNorm.bias",
            "encoder.layer.11.intermediate.dense.bias",
        ]
        num_layer = Converter.get_num_layer(layers)
        self.assertEqual(num_layer, 12)
