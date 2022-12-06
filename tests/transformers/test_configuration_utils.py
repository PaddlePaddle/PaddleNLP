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
from typing import Dict, Optional

from paddlenlp.transformers import RoFormerv2ForTokenClassification
from paddlenlp.transformers.configuration_utils import PretrainedConfig, attribute_map
from paddlenlp.transformers.model_utils import PretrainedModel


class FakeSimplePretrainedModelConfig(PretrainedConfig):
    """simple fake Pretrained Model Config"""

    def __init__(self, a=0, b=1, c=2):
        self.a = a
        self.b = b
        self.c = c


class FakePretrainedModelConfig(PretrainedConfig):
    """Fake Pretrained Model which is similar with actual situation"""

    attribute_map: Dict[str, str] = {
        "num_classes": "num_labels",
    }

    def __init__(self, hidden_dropout_prob: float, **kwargs):
        attribute_map(self, kwargs=kwargs)
        super().__init__(**kwargs)
        self.hidden_dropout_prob = hidden_dropout_prob


class FakeLayer:
    def __init__(self, config: Optional[FakeSimplePretrainedModelConfig] = None, *args, **kwargs):
        super(FakeLayer, self).__init__()

        self.a = config.a
        self.b = config.b
        self.c = config.c


class FakeModel(PretrainedModel):
    def __init__(self, config: FakeSimplePretrainedModelConfig):
        """fake `__init__`, the source of parameters is:
            def __init__(self, model, a, b):
                self.model = model
                self.a = a
                self.b = b
        Args:
            config_or_model (Optional[Union[FakeLayer, FakeSimplePretrainedModelConfig]], optional): config or model instance. Defaults to None.
        """
        super().__init__()

        self.model: FakeLayer = FakeLayer(config)
        self.a = config.a
        self.b = config.b


class ConfigurationUtilsTest(unittest.TestCase):
    def test_parse_config_with_single_config(self):
        # 1. single config
        config = FakeSimplePretrainedModelConfig(a=10, b=11, c=12)
        model = FakeLayer(config)
        assert model.a == 10
        assert model.b == 11

    def test_parse_config_with_full_kwargs(self):
        model = FakeLayer(a=10, b=11, c=12)
        assert model.a == 10
        assert model.b == 11

    def test_parse_config_with_full_args(self):
        model = FakeLayer(10, 11, 12)
        assert model.a == 10
        assert model.b == 11

    def test_parse_config_with_args_and_kwargs(self):
        model = FakeLayer(10, b=11, c=12)
        assert model.a == 10
        assert model.b == 11

    def test_parse_config_and_model_with_single_config(self):
        config = FakeSimplePretrainedModelConfig(a=10, b=11, c=12)
        model = FakeModel(config)
        assert model.a == 10
        assert model.b == 11

    def test_parse_config_and_model_with_model_and_args(self):
        config = FakeSimplePretrainedModelConfig(a=10, b=11, c=12)
        fake_layer = FakeLayer(config)
        model = FakeModel(fake_layer, 100, 110)
        assert model.a == 100
        assert model.b == 110

        assert model.model.a == 10
        assert model.model.b == 11

    def test_parse_config_and_model_with_model_and_kwargs(self):
        config = FakeSimplePretrainedModelConfig(a=10, b=11, c=12)
        fake_layer = FakeLayer(config)
        model = FakeModel(fake_layer, a=100, b=110)

        assert model.a == 100
        assert model.b == 110

        assert model.model.a == 10
        assert model.model.b == 11

    def test_parse_config_and_model_with_model_and_kwargs_and_args(self):
        config = FakeSimplePretrainedModelConfig(a=10, b=11, c=12)
        fake_layer = FakeLayer(config)
        model = FakeModel(fake_layer, 100, b=110)

        assert model.a == 100
        assert model.b == 110

        assert model.model.a == 10
        assert model.model.b == 11

    def test_get_value_with_default_from_config(self):
        config = FakeSimplePretrainedModelConfig(a=10)
        assert config.get("a", None) == 10
        assert config.get("a", None) == config.a
        assert config.get("no_name", 0) == 0


class StandardConfigMappingTest(unittest.TestCase):
    def test_roformer_v2_config(self):
        class FakeRoformerV2(RoFormerv2ForTokenClassification):
            pass

        roformerv2 = FakeRoformerV2.from_pretrained("__internal_testing__/roformerv2")
        hidden_size = roformerv2.get_model_config()["init_args"][0]["hidden_size"]

        FakeRoformerV2.standard_config_map = {"hidden_size": "fake_field"}
        loaded_roformerv2 = FakeRoformerV2.from_pretrained("__internal_testing__/roformerv2")
        fake_field = loaded_roformerv2.get_model_config()["init_args"][0]["fake_field"]
        assert fake_field == hidden_size
