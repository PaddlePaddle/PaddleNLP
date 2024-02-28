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

from paddlenlp.transformers import AutoProcessor, CLIPProcessor
from paddlenlp.utils.log import logger
from tests.testing_utils import slow


class ProcessorLoadTester(unittest.TestCase):
    @parameterized.expand(
        [
            (AutoProcessor, "openai/clip-vit-base-patch32", True, False, False, "./model/hf", None),
            (AutoProcessor, "aistudio/clip-vit-base-patch32", False, True, False, "./model/aistudio", None),
            (CLIPProcessor, "openai/clip-vit-base-patch32", False, False, False, "./model/bos", None),
            (AutoProcessor, "xiaoguailin/clip-vit-large-patch14", False, False, True, "./model/modelscope", None),
            (
                AutoProcessor,
                "aistudio/paddlenlp-test-model",
                False,
                True,
                False,
                "./model/subfolder/aistudio",
                "clip-vit-base-patch32",
            ),
            (
                CLIPProcessor,
                "baicai/paddlenlp-test-model",
                False,
                False,
                False,
                "./model/subfolder/bos",
                "clip-vit-base-patch32",
            ),
        ]
    )
    def test_local(self, processor_cls, model_name, from_hf_hub, from_aistudio, from_modelscope, cache_dir, subfolder):
        logger.info("Download Image processor from local dir")
        if from_modelscope:
            os.environ["from_modelscope"] = "True"
        processor = processor_cls.from_pretrained(
            model_name, from_hf_hub=from_hf_hub, from_aistudio=from_aistudio, cache_dir=cache_dir, subfolder=subfolder
        )
        processor.save_pretrained(cache_dir)
        local_processor = processor_cls.from_pretrained(cache_dir)
        os.environ["from_modelscope"] = "False"

    @parameterized.expand(
        [
            (AutoProcessor, "openai/clip-vit-base-patch32", True, False, False, None),
            (CLIPProcessor, "aistudio/clip-vit-base-patch32", False, True, False, None),
            (AutoProcessor, "openai/clip-vit-base-patch32", False, False, False, None),
            (AutoProcessor, "xiaoguailin/clip-vit-large-patch14", False, False, True, None),
            (CLIPProcessor, "aistudio/paddlenlp-test-model", False, True, False, "clip-vit-base-patch32"),
            (AutoProcessor, "baicai/paddlenlp-test-model", False, False, False, "clip-vit-base-patch32"),
        ]
    )
    def test_download_cache(self, processor_cls, model_name, from_hf_hub, from_aistudio, from_modelscope, subfolder):
        logger.info("Download Image processor from local dir")
        if from_modelscope:
            os.environ["from_modelscope"] = "True"
        processor = processor_cls.from_pretrained(
            model_name, from_hf_hub=from_hf_hub, from_aistudio=from_aistudio, subfolder=subfolder
        )
        local_processor = processor_cls.from_pretrained(
            model_name, from_hf_hub=from_hf_hub, from_aistudio=from_aistudio, subfolder=subfolder
        )
        os.environ["from_modelscope"] = "False"
