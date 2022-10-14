# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import unittest
from paddlenlp.utils.log import logger
from paddlenlp.transformers import AutoTokenizer
import faster_tokenizer
from faster_tokenizer import ErnieFasterTokenizer, models

logger.logger.setLevel('ERROR')


class TestTokenizerJson(unittest.TestCase):

    def setUp(self):
        wordpiece_tokenizer = AutoTokenizer.from_pretrained("ernie-1.0")
        ernie_vocab = wordpiece_tokenizer.vocab.token_to_idx
        self.faster_tokenizer = ErnieFasterTokenizer(ernie_vocab)


class TestNormalizerJson(TestTokenizerJson):

    def check_normalizer_json(self, normalizer):
        self.faster_tokenizer.normalizer = normalizer
        json_file = str(normalizer.__class__) + ".json"
        self.faster_tokenizer.save(json_file)
        tokenizer = ErnieFasterTokenizer.from_file(json_file)
        os.remove(json_file)
        self.assertEqual(normalizer.__getstate__(),
                         tokenizer.normalizer.__getstate__())

    def test_replace(self):
        replace_normalizer = faster_tokenizer.normalizers.ReplaceNormalizer(
            "''", "\"")
        self.check_normalizer_json(replace_normalizer)

    def test_strip(self):
        strip_normalizer = faster_tokenizer.normalizers.StripNormalizer(
            True, True)
        self.check_normalizer_json(strip_normalizer)

    def test_strip_accent(self):
        strip_normalizer = faster_tokenizer.normalizers.StripAccentsNormalizer()
        self.check_normalizer_json(strip_normalizer)

    def test_nfc(self):
        nfc_normalizer = faster_tokenizer.normalizers.NFCNormalizer()
        self.check_normalizer_json(nfc_normalizer)

    def test_nfkc(self):
        nfkc_normalizer = faster_tokenizer.normalizers.NFKCNormalizer()
        self.check_normalizer_json(nfkc_normalizer)

    def test_nfd(self):
        nfd_normalizer = faster_tokenizer.normalizers.NFDNormalizer()
        self.check_normalizer_json(nfd_normalizer)

    def test_nfkd(self):
        nfkd_normalizer = faster_tokenizer.normalizers.NFKDNormalizer()
        self.check_normalizer_json(nfkd_normalizer)

    def test_nmt(self):
        nmt_normalizer = faster_tokenizer.normalizers.NmtNormalizer()
        self.check_normalizer_json(nmt_normalizer)

    def test_lowercase(self):
        lowercase_normalizer = faster_tokenizer.normalizers.LowercaseNormalizer(
        )
        self.check_normalizer_json(lowercase_normalizer)

    def test_sequence(self):
        lowercase_normalizer = faster_tokenizer.normalizers.LowercaseNormalizer(
        )
        sequence_normalizer = faster_tokenizer.normalizers.SequenceNormalizer(
            normalizers=[lowercase_normalizer])
        self.check_normalizer_json(sequence_normalizer)


if __name__ == "__main__":
    unittest.main()
