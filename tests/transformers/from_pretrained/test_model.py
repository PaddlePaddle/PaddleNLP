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

    # bulid-in的时候是获取到url从bos下载，所以只有一个下载源，而且一定是pd权重
    @parameterized.expand(
        [
            # 测试t5，指定不同的下载源（不会生效）
            (AutoModel, "t5-base", True, False, False, None, None, "./model/t5-base"),
            (T5Model, "t5-base", True, False, True, None, None, "./model/t5-base"),
            # 测试bert，指定不同use_safetensors参数（不会生效）
            (BertModel, "bert-base-uncased", False, True, False, True, None, "./model/bert-base-uncased"),
            (AutoModel, "bert-base-uncased", False, True, False, False, None, "./model/bert-base-uncased"),
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
            # hf情况下，use_safetensors默认、false、true的情况
            (T5Model, "Baicai003/tiny-t5", True, False, False, None, None, "./model/hf/tiny-t5"),
            (AutoModel, "Baicai003/tiny-t5", True, False, False, False, None, "./model/hf/tiny-t5"),
            (AutoModel, "Baicai003/tiny-t5", True, False, False, True, None, "./model/hf/tiny-t5"),
            # hf情况下，有subfloder，use_safetensors默认、false、true的情况
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
                AutoModel,
                "Baicai003/paddlenlp-test-model",
                True,
                False,
                False,
                True,
                "tiny-clip-one",
                "./model/hf/t5-base",
            ),
            # bos情况下，use_safetensors默认、false、true的情况
            (CLIPTextModel, "baicai/tiny-clip", False, False, False, None, None, "./model/bos/tiny-clip"),
            (AutoModel, "baicai/tiny-clip", False, False, False, False, None, "./model/bos/tiny-clip"),
            (CLIPTextModel, "baicai/tiny-clip", False, False, False, True, None, "./model/bos/tiny-clip"),
            # bos情况下，有subfloder，use_safetensors默认、false、true的情况
            (
                CLIPTextModel,
                "baicai/paddlenlp-test-model",
                False,
                False,
                False,
                None,
                "tiny-clip",
                "./model/bos/tiny-clip",
            ),
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
            # aistudio情况下，use_safetensors默认、false、true的情况
            (AutoModel, "aistudio/tiny-clip", False, True, False, None, None, "./model/aistudio/tiny-clip"),
            (CLIPTextModel, "aistudio/tiny-clip", False, True, False, False, None, "./model/aistudio/tiny-clip"),
            (AutoModel, "aistudio/tiny-clip", False, True, False, True, None, "./model/aistudio/tiny-clip"),
            # aistudio情况下，有subfloder，use_safetensors默认、false、true的情况
            (
                CLIPTextModel,
                "aistudio/paddlenlp-test-model",
                False,
                True,
                False,
                None,
                "tiny-clip",
                "./model/aistudio/tiny-clip",
            ),
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
            # modelscope情况下，use_safetensors默认、false、true的情况
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
            (
                CLIPTextModel,
                "xiaoguailin/clip-vit-large-patch14",
                False,
                False,
                True,
                True,
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
            # hf情况下，use_safetensors默认、false、true的情况
            (T5Model, "Baicai003/tiny-t5", True, False, False, None, None, "./model/hf/tiny-t5"),
            (AutoModel, "Baicai003/tiny-t5", True, False, False, False, None, "./model/hf/tiny-t5"),
            (AutoModel, "Baicai003/tiny-t5", True, False, False, True, None, "./model/hf/tiny-t5"),
            # hf情况下，有subfolder，use_safetensors默认、false、true的情况
            (CLIPTextModel, "Baicai003/paddlenlp-test-model", True, False, False, None, "tiny-clip-one"),
            (AutoModel, "Baicai003/paddlenlp-test-model", True, False, False, False, "tiny-clip-one"),
            (CLIPTextModel, "Baicai003/paddlenlp-test-model", True, False, False, True, "tiny-clip-one"),
            # bos情况下，use_safetensors默认、false、true的情况
            (AutoModel, "baicai/tiny-clip", False, False, False, None, None),
            (CLIPTextModel, "baicai/tiny-clip", False, False, False, True, None),
            (AutoModel, "baicai/tiny-clip", False, False, False, False, None),
            # bos情况下，有subfolder，use_safetensors默认、false、true的情况
            (CLIPTextModel, "baicai/paddlenlp-test-model", False, False, False, None, "tiny-clip"),
            (AutoModel, "baicai/paddlenlp-test-model", False, False, False, False, "tiny-clip"),
            (CLIPTextModel, "baicai/paddlenlp-test-model", False, False, False, True, "tiny-clip"),
            # aistudio情况下，use_safetensors默认、true和false的情况
            (AutoModel, "aistudio/tiny-clip", False, True, False, None, None),
            (CLIPTextModel, "aistudio/tiny-clip", False, True, False, True, None),
            (AutoModel, "aistudio/tiny-clip", False, True, False, False, None),
            #  aistudio情况下，有subfolder，use_safetensors默认、false、true的情况
            (CLIPTextModel, "aistudio/paddlenlp-test-model", False, True, False, None, "tiny-clip"),
            (AutoModel, "aistudio/paddlenlp-test-model", False, True, False, False, "tiny-clip"),
            (CLIPTextModel, "aistudio/paddlenlp-test-model", False, True, False, True, "tiny-clip"),
            # modelscope情况下，use_safetensors默认、true和false的情况
            (CLIPTextModel, "xiaoguailin/clip-vit-large-patch14", False, False, True, None, None),
            (AutoModel, "xiaoguailin/clip-vit-large-patch14", False, False, True, False, None),
            (CLIPTextModel, "xiaoguailin/clip-vit-large-patch14", False, False, True, True, None),
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
