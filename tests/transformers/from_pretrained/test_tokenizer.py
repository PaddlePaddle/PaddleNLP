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

from paddlenlp.transformers import AutoTokenizer, T5Tokenizer
from paddlenlp.utils.log import logger


class TokenizerLoadTester(unittest.TestCase):

    # 这是内置的是下载哪些文件
    @parameterized.expand(
        [
            (T5Tokenizer, "t5-small", True, False, False),
            (AutoTokenizer, "t5-small", True, False, False),
            (T5Tokenizer, "AI-ModelScope/t5-base", False, False, True),
        ]
    )
    def test_build_in(self, tokenizer_cls, model_name, from_hf_hub, from_aistudio, from_modelscope):
        logger.info("Load tokenizer from build-in dict")
        if from_modelscope:
            os.environ["from_modelscope"] = "True"
        tokenizer_cls.from_pretrained(model_name, from_hf_hub=from_hf_hub, from_aistudio=from_aistudio)
        os.environ["from_modelscope"] = "False"

    @parameterized.expand(
        [
            (T5Tokenizer, "t5-small", True, False, False, "./paddlenlp-test-tokenizer-hf"),
            (AutoTokenizer, "aistudio/t5-small", False, True, False, "./paddlenlp-test-tokenizer-aistudio"),
            (AutoTokenizer, "t5-small", False, False, False, "./paddlenlp-test-tokenizer-bos"),
            (T5Tokenizer, "langboat/mengzi-t5-base", False, False, True, "./paddlenlp-test-tokenizer-modelscope"),
        ]
    )
    def test_local(self, tokenizer_cls, model_name, from_hf_hub, from_aistudio, from_modelscope, cache_dir):
        logger.info("Download tokenizer from local dir")
        if from_modelscope:
            os.environ["from_modelscope"] = "True"
        tokenizer = tokenizer_cls.from_pretrained(
            model_name, from_hf_hub=from_hf_hub, from_aistudio=from_aistudio, cache_dir=cache_dir
        )
        tokenizer.save_pretrained(cache_dir)
        local_tokenizer = tokenizer_cls.from_pretrained(cache_dir)
        assert tokenizer("PaddleNLP is a better project") == local_tokenizer("PaddleNLP is a better project")
        os.environ["from_modelscope"] = "False"

    @parameterized.expand(
        [
            (T5Tokenizer, "Baicai003/paddlenlp-test-model", True, False, False, "t5-small"),
            (T5Tokenizer, "aistudio/paddlenlp-test-model", False, True, False, "t5-small"),
            (AutoTokenizer, "baicai/paddlenlp-test-model", False, False, False, "t5-small"),
            (T5Tokenizer, "langboat/mengzi-t5-base", False, False, True, None),
            (T5Tokenizer, "langboat/mengzi-t5-base-mt", False, False, True, ""),
            # roberta
            (AutoTokenizer, "roberta-base", True, False, False, ""),
            (AutoTokenizer, "roberta-base", False, False, False, ""),
            (AutoTokenizer, "roberta-base", False, False, True, ""),
        ]
    )
    def test_download_cache(self, tokenizer_cls, model_name, from_hf_hub, from_aistudio, from_modelscope, subfolder):
        logger.info("Download tokenizer from different sources with subfolder")
        if from_modelscope:
            os.environ["from_modelscope"] = "True"
            assert subfolder is None or subfolder == ""
        tokenizer = tokenizer_cls.from_pretrained(
            model_name, subfolder=subfolder, from_hf_hub=from_hf_hub, from_aistudio=from_aistudio
        )
        auto_tokenizer = AutoTokenizer.from_pretrained(
            model_name, subfolder=subfolder, from_hf_hub=from_hf_hub, from_aistudio=from_aistudio
        )
        assert tokenizer("PaddleNLP is a better project") == auto_tokenizer("PaddleNLP is a better project")
        os.environ["from_modelscope"] = "False"
