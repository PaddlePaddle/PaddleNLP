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

from paddlenlp.transformers import AutoImageProcessor, CLIPImageProcessor
from paddlenlp.utils.log import logger


class ImageProcessorLoadTester(unittest.TestCase):
    @parameterized.expand(
        [
            (AutoImageProcessor, "openai/clip-vit-base-patch32", True, False, False, "./model/hf", None),
            (AutoImageProcessor, "aistudio/clip-vit-base-patch32", False, True, False, "./model/aistudio", None),
            (CLIPImageProcessor, "openai/clip-vit-base-patch32", False, False, False, "./model/bos", None),
            (AutoImageProcessor, "thomas/clip-vit-base-patch32", False, False, True, "./model/modelscope", None),
            (
                AutoImageProcessor,
                "aistudio/paddlenlp-test-model",
                False,
                True,
                False,
                "./model/subfolder/aistudio",
                "clip-vit-base-patch32",
            ),
            (
                CLIPImageProcessor,
                "baicai/paddlenlp-test-model",
                False,
                False,
                False,
                "./model/subfolder/bos",
                "clip-vit-base-patch32",
            ),
        ]
    )
    def test_local(
        self, image_processor_cls, model_name, from_hf_hub, from_aistudio, from_modelscope, cache_dir, subfolder
    ):
        logger.info("Download Image processor from local dir")
        if from_modelscope:
            os.environ["from_modelscope"] = "True"
        image_processor = image_processor_cls.from_pretrained(
            model_name, from_hf_hub=from_hf_hub, from_aistudio=from_aistudio, cache_dir=cache_dir, subfolder=subfolder
        )
        image_processor.save_pretrained(cache_dir)
        image_processor_cls.from_pretrained(cache_dir)
        os.environ["from_modelscope"] = "False"

    @parameterized.expand(
        [
            (AutoImageProcessor, "openai/clip-vit-base-patch32", True, False, False, None),
            (CLIPImageProcessor, "aistudio/clip-vit-base-patch32", False, True, False, None),
            (AutoImageProcessor, "openai/clip-vit-base-patch32", False, False, False, None),
            (AutoImageProcessor, "thomas/clip-vit-base-patch32", False, False, True, None),
            (CLIPImageProcessor, "aistudio/paddlenlp-test-model", False, True, False, "clip-vit-base-patch32"),
            (AutoImageProcessor, "baicai/paddlenlp-test-model", False, False, False, "clip-vit-base-patch32"),
        ]
    )
    def test_download_cache(
        self, image_processor_cls, model_name, from_hf_hub, from_aistudio, from_modelscope, subfolder
    ):
        logger.info("Download Image processor from local dir")
        if from_modelscope:
            os.environ["from_modelscope"] = "True"
        image_processor_cls.from_pretrained(
            model_name, from_hf_hub=from_hf_hub, from_aistudio=from_aistudio, subfolder=subfolder
        )
        image_processor_cls.from_pretrained(
            model_name, from_hf_hub=from_hf_hub, from_aistudio=from_aistudio, subfolder=subfolder
        )
        os.environ["from_modelscope"] = "False"
