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

import os
import shutil
import tempfile
import unittest
from typing import Dict, Optional

from paddlenlp.transformers import BertConfig
from paddlenlp.transformers.configuration_utils import PretrainedConfig, attribute_map
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.utils import CONFIG_NAME
from paddlenlp.utils.env import LEGACY_CONFIG_NAME


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


class ConfigurationUtilsTest:
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
    def test_bert_config_mapping(self):
        # create new fake-bert class to prevent static-attributed modified by this test
        class FakeBertConfig(BertConfig):
            pass

        config = FakeBertConfig.from_pretrained("__internal_testing__/bert")
        hidden_size = config.hidden_size

        FakeBertConfig.attribute_map = {"fake_field": "hidden_size"}

        loaded_config = FakeBertConfig.from_pretrained("__internal_testing__/bert")
        fake_field = loaded_config.fake_field
        self.assertEqual(fake_field, hidden_size)

    def test_from_pretrained_cache_dir(self):
        model_id = "__internal_testing__/tiny-random-bert"
        with tempfile.TemporaryDirectory() as tempdir:
            BertConfig.from_pretrained(model_id, cache_dir=tempdir)
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_id, CONFIG_NAME)))
            # check against double appending model_name in cache_dir
            self.assertFalse(os.path.exists(os.path.join(tempdir, model_id, model_id)))

    def test_load_from_hf(self):
        """test load config from hf"""
        config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-BertModel", from_hf_hub=True)
        self.assertEqual(config.hidden_size, 32)

        with tempfile.TemporaryDirectory() as tempdir:
            config.save_pretrained(tempdir)

            self.assertTrue(os.path.exists(os.path.join(tempdir, CONFIG_NAME)))

            loaded_config = BertConfig.from_pretrained(tempdir)
            self.assertEqual(loaded_config.hidden_size, 32)

    def test_config_mapping(self):
        # create new fake-bert class to prevent static-attributed modified by this test
        class FakeBertConfig(BertConfig):
            pass

        with tempfile.TemporaryDirectory() as tempdir:
            config = FakeBertConfig.from_pretrained("bert-base-uncased")
            config.save_pretrained(tempdir)

            # rename `config.json` -> `model_config.json`
            shutil.move(os.path.join(tempdir, CONFIG_NAME), os.path.join(tempdir, LEGACY_CONFIG_NAME))

            FakeBertConfig.attribute_map = {"fake_field": "hidden_size"}

            loaded_config = FakeBertConfig.from_pretrained(tempdir)
            self.assertEqual(loaded_config.fake_field, config.hidden_size)
