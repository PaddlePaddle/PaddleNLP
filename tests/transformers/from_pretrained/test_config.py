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

import os
import unittest

from parameterized import parameterized

from paddlenlp.transformers import AutoConfig, BertConfig
from paddlenlp.transformers.bloom.configuration import BloomConfig
from paddlenlp.utils.log import logger


class ConfigLoadTester(unittest.TestCase):
    @parameterized.expand(
        [
            (BertConfig, "bert-base-uncased", False, True, False, "vocab_size", 30522),
            (AutoConfig, "bert-base-uncased", True, False, False, "vocab_size", 30522),
        ]
    )
    def test_build_in(
        self, config_cls, model_name, from_hf_hub, from_aistudio, from_modelscope, check_key, check_value
    ):
        logger.info("Load Config from build-in dict")
        if from_modelscope:
            os.environ["from_modelscope"] = "True"
        config = config_cls.from_pretrained(model_name, from_hf_hub=from_hf_hub, from_aistudio=from_aistudio)
        assert config[check_key] == check_value
        os.environ["from_modelscope"] = "False"

    @parameterized.expand(
        [
            (
                BertConfig,
                "bert-base-uncased",
                False,
                True,
                False,
                "./paddlenlp-test-config/bert-base-uncased",
                "hidden_dropout_prob",
            ),
            (
                AutoConfig,
                "bert-base-uncased",
                True,
                False,
                False,
                "./paddlenlp-test-config/bert-base-uncased_2",
                "hidden_dropout_prob",
            ),
        ]
    )
    def test_local(self, config_cls, model_name, from_hf_hub, from_aistudio, from_modelscope, cache_dir, check_key):
        logger.info("Download config from local dir")
        if from_modelscope:
            os.environ["from_modelscope"] = "True"
        config = config_cls.from_pretrained(
            model_name, from_hf_hub=from_hf_hub, from_aistudio=from_aistudio, cache_dir=cache_dir
        )
        config.save_pretrained(cache_dir)
        local_config = config_cls.from_pretrained(cache_dir)
        assert config[check_key] == local_config[check_key]
        os.environ["from_modelscope"] = "False"

    @parameterized.expand(
        [
            (BertConfig, "Baicai003/paddlenlp-test-model", True, False, False, "tiny-bert"),
            (BertConfig, "baicai/paddlenlp-test-model", False, False, False, "tiny-bert"),
            (BertConfig, "aistudio/paddlenlp-test-model", False, True, False, "tiny-bert"),
            (BloomConfig, "bigscience/bloom-7b1", True, False, False, None),
            (BloomConfig, "bigscience/bloom-7b1", False, False, False, None),
            (BertConfig, "langboat/mengzi-bert-base", False, False, True, ""),
            (BertConfig, "langboat/mengzi-bert-base-fin", False, False, True, None),
        ]
    )
    def test_download_cache(self, config_cls, model_name, from_hf_hub, from_aistudio, from_modelscope, subfolder):
        logger.info("Download Config from different sources with subfolder")
        if from_modelscope:
            os.environ["from_modelscope"] = "True"
            assert subfolder is None or subfolder == ""
        config = config_cls.from_pretrained(
            model_name, subfolder=subfolder, from_hf_hub=from_hf_hub, from_aistudio=from_aistudio
        )
        auto_config = AutoConfig.from_pretrained(
            model_name, subfolder=subfolder, from_hf_hub=from_hf_hub, from_aistudio=from_aistudio
        )
        assert config == auto_config
        os.environ["from_modelscope"] = "False"
