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
from typing import Optional, Union, Dict
from paddlenlp.transformers.configuration_utils import attribute_map, parse_config, PretrainedConfig
from paddlenlp.transformers.model_utils import PretrainedModel


class FakeSimplePretrainedModelConfig(PretrainedConfig):
    """simple fake Pretrained Model Config
    """

    def __init__(self, a=0, b=1, c=2):
        self.a = a
        self.b = b
        self.c = c


class FakePretrainedModelConfig(PretrainedConfig):
    """Fake Pretrained Model which is similar with actual situation
    """
    attribute_map: Dict[str, str] = {
        "num_classes": "num_labels",
    }

    def __init__(self, hidden_dropout_prob: float, **kwargs):
        attribute_map(self, kwargs=kwargs)
        super().__init__(**kwargs)
        self.hidden_dropout_prob = hidden_dropout_prob


class FakeLayer:

    def __init__(self,
                 config: Optional[FakeSimplePretrainedModelConfig] = None,
                 *args,
                 **kwargs):
        super(FakeLayer, self).__init__()

        config: FakeSimplePretrainedModelConfig = parse_config(
            config_or_model=config,
            config_class=FakeSimplePretrainedModelConfig,
            args=args,
            kwargs=kwargs,
            fields=["a", "b", "c"],
        )

        self.a = config.a
        self.b = config.b
        self.c = config.c


class FakeModel(PretrainedModel):

    def __init__(self,
                 config_or_model: Optional[Union[
                     FakeLayer, FakeSimplePretrainedModelConfig]] = None,
                 *args,
                 **kwargs):
        """fake `__init__`, the source of parameters is:

            def __init__(self, model, a, b):
                self.model = model
                self.a = a
                self.b = b

        Args:
            config_or_model (Optional[Union[FakeLayer, FakeSimplePretrainedModelConfig]], optional): config or model instance. Defaults to None.
        """
        super().__init__()

        config, model = parse_config(
            config_or_model=config_or_model,
            config_class=FakeSimplePretrainedModelConfig,
            args=args,
            kwargs=kwargs,
            fields=["a", ("b", 2)],
        )

        self.model: FakeLayer = model
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
