# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.transformers import AutoModel, BertModel, CLIPModel, T5Model
from paddlenlp.utils.log import logger


class ModelLoadTester(unittest.TestCase):
    def test_bert_load(self):
        logger.info("Download model from PaddleNLP BOS")
        bert_model = BertModel.from_pretrained("bert-base-uncased", from_hf_hub=False)
        bert_model = AutoModel.from_pretrained("bert-base-uncased", from_hf_hub=False)

        logger.info("Download model from PaddleNLP BOS with subfolder")
        bert_model = BertModel.from_pretrained(
            "baicai/paddlenlp-test-model", subfolder="bert-base-uncased", from_hf_hub=False
        )
        bert_model = AutoModel.from_pretrained(
            "baicai/paddlenlp-test-model", subfolder="bert-base-uncased", from_hf_hub=False
        )

        logger.info("Download model from aistudio")
        bert_model = BertModel.from_pretrained("aistudio/bert-base-uncased", from_aistudio=True)
        bert_model = AutoModel.from_pretrained("aistudio/bert-base-uncased", from_aistudio=True)

        logger.info("Download model from aistudio with subfolder")
        bert_model = BertModel.from_pretrained(
            "aistudio/paddlenlp-test-model", subfolder="bert-base-uncased", from_aistudio=True
        )
        bert_model = AutoModel.from_pretrained(
            "aistudio/paddlenlp-test-model", subfolder="bert-base-uncased", from_aistudio=True
        )

        logger.info("Download model from local")
        bert_model.save_pretrained("./paddlenlp-test-model/bert-base-uncased")
        bert_model = BertModel.from_pretrained("./paddlenlp-test-model/", subfolder="bert-base-uncased")
        bert_model = AutoModel.from_pretrained("./paddlenlp-test-model/", subfolder="bert-base-uncased")

    def test_clip_load(self):
        logger.info("Download model from PaddleNLP BOS")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", from_hf_hub=False)
        clip_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32", from_hf_hub=False)

        logger.info("Download model from PaddleNLP BOS with subfolder")
        clip_model = CLIPModel.from_pretrained(
            "baicai/paddlenlp-test-model", subfolder="clip-vit-base-patch32", from_hf_hub=False
        )
        clip_model = AutoModel.from_pretrained(
            "baicai/paddlenlp-test-model", subfolder="clip-vit-base-patch32", from_hf_hub=False
        )

        logger.info("Download model from aistudio")
        clip_model = CLIPModel.from_pretrained("aistudio/clip-vit-base-patch32", from_aistudio=True)
        clip_model = AutoModel.from_pretrained("aistudio/clip-vit-base-patch32", from_aistudio=True)

        logger.info("Download model from aistudio with subfolder")
        clip_model = CLIPModel.from_pretrained(
            "aistudio/paddlenlp-test-model", subfolder="clip-vit-base-patch32", from_aistudio=True
        )
        clip_model = AutoModel.from_pretrained(
            "aistudio/paddlenlp-test-model", subfolder="clip-vit-base-patch32", from_aistudio=True
        )

        logger.info("Download model from local")
        clip_model.save_pretrained("./paddlenlp-test-model/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("./paddlenlp-test-model/", subfolder="clip-vit-base-patch32")
        clip_model = AutoModel.from_pretrained("./paddlenlp-test-model/", subfolder="clip-vit-base-patch32")

    def test_t5_load(self):
        logger.info("Download model from PaddleNLP BOS")
        t5_model = T5Model.from_pretrained("t5-small", from_hf_hub=False)
        t5_model = AutoModel.from_pretrained("t5-small", from_hf_hub=False)

        logger.info("Download model from PaddleNLP BOS with subfolder")
        t5_model = T5Model.from_pretrained("baicai/paddlenlp-test-model", subfolder="t5-small", from_hf_hub=False)
        t5_model = AutoModel.from_pretrained("baicai/paddlenlp-test-model", subfolder="t5-small", from_hf_hub=False)

        logger.info("Download model from aistudio")
        t5_model = T5Model.from_pretrained("aistudio/t5-small", from_aistudio=True)
        t5_model = AutoModel.from_pretrained("aistudio/t5-small", from_aistudio=True)

        logger.info("Download model from aistudio with subfolder")
        t5_model = T5Model.from_pretrained("aistudio/paddlenlp-test-model", subfolder="t5-small", from_aistudio=True)
        t5_model = AutoModel.from_pretrained("aistudio/paddlenlp-test-model", subfolder="t5-small", from_aistudio=True)

        logger.info("Download model from local")
        t5_model.save_pretrained("./paddlenlp-test-model/t5-small")
        t5_model = T5Model.from_pretrained("./paddlenlp-test-model/", subfolder="t5-small")
        t5_model = AutoModel.from_pretrained("./paddlenlp-test-model/", subfolder="t5-small")
