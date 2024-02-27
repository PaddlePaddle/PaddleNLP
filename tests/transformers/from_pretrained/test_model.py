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

import pytest
from parameterized import parameterized

from paddlenlp.transformers import AutoModel, BertModel, CLIPTextModel, T5Model
from paddlenlp.utils.log import logger


class ModelLoadTester(unittest.TestCase):
    @pytest.mark.skip
    def test_config_diff(self, config_1, config_2):
        config_1 = config_1.to_dict()
        config_2 = config_2.to_dict()
        config_1.pop("architectures", None)
        config_2.pop("architectures", None)
        assert config_1 == config_2, "config not equal"

    # 获得模型url，直接进行下载
    @parameterized.expand(
        [
            (BertModel, "bert-base-uncased", False, True, False, True, None, "./model/bert-base-uncased"),
            (AutoModel, "t5-base", True, False, False, None, None, "./model/t5-base"),
            (AutoModel, "t5-base", True, False, True, None, None, "./model/t5-base"),
            (BertModel, "bert-base-uncased", False, True, False, False, None, "./model/bert-base-uncased"),
        ]
    )
    def test_bulid_in(
        self, model_cls, model_name, from_hf_hub, from_aistudio, from_modelscope, use_safetensors, subfolder, cache_dir
    ):
        logger.info("Download model from build-in url")
        if from_modelscope:
            os.environ["from_modelscope"] = "True"
        model_cls.from_pretrained(
            model_name,
            from_hf_hub=from_hf_hub,
            from_aistudio=from_aistudio,
            use_safetensors=use_safetensors,
            subfolder=subfolder,
            cache_dir=cache_dir,
        )
        os.environ["from_modelscope"] = "False"

    @parameterized.expand(
        [
            (T5Model, "t5-base", True, False, False, None, None, "./model/hf/t5-base"),
            (AutoModel, "t5-base", True, False, False, False, None, "./model/hf/t5-base"),
            (
                AutoModel,
                "Baicai003/paddlenlp-test-model",
                True,
                False,
                False,
                False,
                "tiny-clip-one",
                "./model/hf/t5-base",
            ),
            (
                CLIPTextModel,
                "Baicai003/paddlenlp-test-model",
                True,
                False,
                False,
                None,
                "tiny-clip-one",
                "./model/hf/t5-base",
            ),
            (CLIPTextModel, "baicai/tiny-clip", False, False, False, True, None, "./model/bos/tiny-clip"),
            (AutoModel, "baicai/tiny-clip", False, False, False, False, None, "./model/bos/tiny-clip"),
            (
                AutoModel,
                "baicai/paddlenlp-test-model",
                False,
                False,
                False,
                False,
                "tiny-clip",
                "./model/bos/tiny-clip",
            ),
            (
                CLIPTextModel,
                "baicai/paddlenlp-test-model",
                False,
                False,
                False,
                True,
                "tiny-clip",
                "./model/bos/tiny-clip",
            ),
            (CLIPTextModel, "aistudio/tiny-clip", False, True, False, True, None, "./model/aistudio/tiny-clip"),
            (AutoModel, "aistudio/tiny-clip", False, True, False, False, None, "./model/aistudio/tiny-clip"),
            (
                AutoModel,
                "aistudio/paddlenlp-test-model",
                False,
                True,
                False,
                False,
                "tiny-clip",
                "./model/aistudio/tiny-clip",
            ),
            (
                CLIPTextModel,
                "aistudio/paddlenlp-test-model",
                False,
                True,
                False,
                True,
                "tiny-clip",
                "./model/aistudio/tiny-clip",
            ),
            (
                CLIPTextModel,
                "xiaoguailin/clip-vit-large-patch14",
                False,
                False,
                True,
                None,
                None,
                "./model/modelscope/clip-vit",
            ),
            (
                AutoModel,
                "xiaoguailin/clip-vit-large-patch14",
                False,
                False,
                True,
                False,
                None,
                "./model/modelscope/clip-vit",
            ),
        ]
    )
    def test_local(
        self, model_cls, model_name, from_hf_hub, from_aistudio, from_modelscope, use_safetensors, subfolder, cache_dir
    ):
        if from_modelscope:
            os.environ["from_modelscope"] = "True"
        model = model_cls.from_pretrained(
            model_name,
            from_hf_hub=from_hf_hub,
            from_aistudio=from_aistudio,
            use_safetensors=use_safetensors,
            subfolder=subfolder,
            cache_dir=cache_dir,
        )
        model.save_pretrained(cache_dir)
        local_model = model_cls.from_pretrained(cache_dir)
        self.test_config_diff(model.config, local_model.config)
        os.environ["from_modelscope"] = "False"

    @parameterized.expand(
        [
            (T5Model, "t5-base", True, False, False, None, None),
            (AutoModel, "t5-base", True, False, False, False, None),
            (AutoModel, "Baicai003/paddlenlp-test-model", True, False, False, False, "tiny-clip-one"),
            (CLIPTextModel, "Baicai003/paddlenlp-test-model", True, False, False, None, "tiny-clip-one"),
            (CLIPTextModel, "baicai/tiny-clip", False, False, False, True, None),
            (AutoModel, "baicai/tiny-clip", False, False, False, False, None),
            (AutoModel, "baicai/paddlenlp-test-model", False, False, False, False, "tiny-clip"),
            (CLIPTextModel, "baicai/paddlenlp-test-model", False, False, False, True, "tiny-clip"),
            (CLIPTextModel, "aistudio/tiny-clip", False, True, False, True, None),
            (AutoModel, "aistudio/tiny-clip", False, True, False, False, None),
            (AutoModel, "aistudio/paddlenlp-test-model", False, True, False, False, "tiny-clip"),
            (CLIPTextModel, "aistudio/paddlenlp-test-model", False, True, False, True, "tiny-clip"),
            (CLIPTextModel, "xiaoguailin/clip-vit-large-patch14", False, False, True, None, None),
            (AutoModel, "xiaoguailin/clip-vit-large-patch14", False, False, True, False, None),
        ]
    )
    def test_download_cache(
        self, model_cls, model_name, from_hf_hub, from_aistudio, from_modelscope, use_safetensors, subfolder
    ):
        if from_modelscope:
            os.environ["from_modelscope"] = "True"
        model = model_cls.from_pretrained(
            model_name,
            from_hf_hub=from_hf_hub,
            from_aistudio=from_aistudio,
            use_safetensors=use_safetensors,
            subfolder=subfolder,
        )
        local_model = model_cls.from_pretrained(
            model_name,
            from_hf_hub=from_hf_hub,
            from_aistudio=from_aistudio,
            use_safetensors=use_safetensors,
            subfolder=subfolder,
        )
        self.test_config_diff(model.config, local_model.config)
        os.environ["from_modelscope"] = "False"
