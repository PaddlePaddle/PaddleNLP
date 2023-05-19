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
import os
import unittest
from tempfile import TemporaryDirectory

from paddlenlp.dataaug import (
    SentenceBackTranslate,
    SentenceContinue,
    SentenceGenerate,
    SentenceSummarize,
)
from paddlenlp.transformers import AutoModelForConditionalGeneration, AutoTokenizer


class TestSentAug(unittest.TestCase):
    def setUp(self):
        self.sequences = ["人类语言是抽象的信息符号。", "而计算机只能处理数值化的信息。"]
        self.max_length = 3

    def test_sent_generate(self):
        aug = SentenceGenerate(model_name="__internal_testing__/tiny-random-roformer-sim", max_length=self.max_length)
        augmented = aug.augment(self.sequences)
        self.assertEqual(len(self.sequences), len(augmented))
        self.assertEqual(aug.create_n, len(augmented[0]))
        self.assertEqual(aug.create_n, len(augmented[1]))

    def test_sent_summarize(self):
        model = AutoModelForConditionalGeneration.from_pretrained(
            "__internal_testing__/tiny-random-mbart", max_length=self.max_length
        )
        tokenizer = AutoTokenizer.from_pretrained("__internal_testing__/tiny-random-mbart")
        model_path = os.path.join(TemporaryDirectory().name, "model")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        aug = SentenceSummarize(task_path=model_path)
        augmented = aug.augment(self.sequences)
        self.assertEqual(len(self.sequences), len(augmented))
        self.assertEqual(aug.create_n, len(augmented[0]))
        self.assertEqual(aug.create_n, len(augmented[1]))

    def test_sent_backtranslate(self):
        aug = SentenceBackTranslate(
            from_model_name="__internal_testing__/tiny-random-mbart",
            to_model_name="__internal_testing__/tiny-random-mbart",
            max_length=self.max_length,
        )
        augmented = aug.augment(self.sequences)
        self.assertEqual(len(self.sequences), len(augmented))
        self.assertEqual(1, len(augmented[0]))
        self.assertEqual(1, len(augmented[1]))

    def test_sent_continue(self):
        aug = SentenceContinue(model_name="__internal_testing__/tiny-random-gpt", max_length=self.max_length)
        augmented = aug.augment(self.sequences)
        self.assertEqual(len(self.sequences), len(augmented))
        self.assertEqual(aug.create_n, len(augmented[0]))
        self.assertEqual(aug.create_n, len(augmented[1]))


if __name__ == "__main__":
    unittest.main()
