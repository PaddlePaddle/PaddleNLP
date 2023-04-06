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

from __future__ import annotations

import hashlib
import importlib
import os
import unittest
from tempfile import TemporaryDirectory


class LockFileTest(unittest.TestCase):
    test_url = (
        "https://bj.bcebos.com/paddlenlp/models/transformers/roformerv2/roformer_v2_chinese_char_small/vocab.txt"
    )

    def test_downloading_with_exist_file(self):

        from paddlenlp.utils.downloader import get_path_from_url_with_filelock

        with TemporaryDirectory() as tempdir:
            lock_file_name = hashlib.md5((self.test_url + tempdir).encode("utf-8")).hexdigest()
            lock_file_path = os.path.join(tempdir, ".lock", lock_file_name)
            os.makedirs(os.path.dirname(lock_file_path), exist_ok=True)

            # create lock file
            with open(lock_file_path, "w", encoding="utf-8") as f:
                f.write("temp test")

            # downloading with exist lock file
            config_file = get_path_from_url_with_filelock(self.test_url, root_dir=tempdir)
            self.assertIsNotNone(config_file)

    def test_downloading_with_opened_exist_file(self):

        from paddlenlp.utils.downloader import get_path_from_url_with_filelock

        with TemporaryDirectory() as tempdir:
            lock_file_name = hashlib.md5((self.test_url + tempdir).encode("utf-8")).hexdigest()
            lock_file_path = os.path.join(tempdir, ".lock", lock_file_name)
            os.makedirs(os.path.dirname(lock_file_path), exist_ok=True)

            # create lock file
            with open(lock_file_path, "w", encoding="utf-8") as f:
                f.write("temp test")

            # downloading with opened lock file
            open_mode = os.O_RDWR | os.O_CREAT | os.O_TRUNC
            _ = os.open(lock_file_path, open_mode)
            config_file = get_path_from_url_with_filelock(self.test_url, root_dir=tempdir)
            self.assertIsNotNone(config_file)


class DynamicCommunityTest(unittest.TestCase):
    # only exist in dynamic community test directory
    model_name = "__internal_testing__/bert-dynamic-community"
    download_related_modules = [
        "paddlenlp.utils.downloader",
        "paddlenlp.transformers.model_utils",
        "paddlenlp.transformers.configuration_utils",
        "paddlenlp.transformers.auto.modeling",
        "paddlenlp.transformers.auto.configuration",
    ]

    def _refresh_community_url(self, value):
        for module_name in self.download_related_modules:
            module = importlib.import_module(module_name)
            module.COMMUNITY_MODEL_PREFIX = value

    def setUp(self) -> None:
        download_module = importlib.import_module("paddlenlp.utils.downloader")
        self.old_url = download_module.COMMUNITY_MODEL_PREFIX

        os.environ["COMMUNITY_MODEL_NAME"] = "https://bj.bcebos.com/paddlenlp/models/community_test"
        new_url = os.getenv("COMMUNITY_MODEL_NAME", "community")
        self._refresh_community_url(new_url)

    def tearDown(self) -> None:
        os.environ.pop("COMMUNITY_MODEL_NAME", None)
        self._refresh_community_url(self.old_url)

    def test_bert_init(self):
        from paddlenlp.transformers import BertModel

        model = BertModel.from_pretrained(self.model_name)
        self.assertIsNotNone(model)

    def test_community_url(self):
        from paddlenlp.utils.downloader import (
            COMMUNITY_MODEL_PREFIX,
            is_url,
            url_file_exists,
        )

        # copy from: https://github.com/PaddlePaddle/PaddleNLP/blob/1c8a4f13bc9ad5ad9e2aa3fc9b5e844d39a66977/paddlenlp/transformers/model_utils.py#L706

        community_model_file_path = "/".join([COMMUNITY_MODEL_PREFIX, self.model_name, "model_state.pdparams"])
        self.assertTrue(is_url(community_model_file_path))
        self.assertTrue(url_file_exists(community_model_file_path))

    def test_auto_from_pretrained(self):
        from paddlenlp.transformers import AutoConfig, AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(self.model_name)
        self.assertIsNotNone(model)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.assertIsNotNone(tokenizer)

        config = AutoConfig.from_pretrained(self.model_name)
        self.assertIsNotNone(config)
