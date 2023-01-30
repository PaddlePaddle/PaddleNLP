# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from parameterized import parameterized

from paddlenlp.prompt import ManualVerbalizer, SoftVerbalizer
from paddlenlp.transformers import AutoModelForMaskedLM, AutoTokenizer


class ManualVerbalizerTest(unittest.TestCase):
    """
    Unittest for ManualVerbalizer
    """

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("__internal_testing__/tiny-random-ernie")
        cls.default_label_words = {"正向": "很", "负向": "不"}
        cls.kwargs = {
            "token_aggregate_type": "first",
            "word_aggregate_type": "first",
            "mask_aggregate_type": "first",
            "post_log_softmax": False,
        }
        cls.save_path = "_tmp_test_saved_verbalizer"
        cls.default_verb = ManualVerbalizer(label_words=cls.default_label_words, tokenizer=cls.tokenizer)

    def test_kwargs(self):
        verb = ManualVerbalizer(tokenizer=self.tokenizer, label_words=self.default_label_words, **self.kwargs)
        self.assertEqual(verb.token_aggregate_type, "first")
        self.assertEqual(verb.word_aggregate_type, "first")
        self.assertEqual(verb.mask_aggregate_type, "first")
        self.assertFalse(verb.post_log_softmax)

    @parameterized.expand(
        [
            (
                {"0": "负向", "1": "正向"},
                ["0", "1"],
                {"0": ["负向"], "1": ["正向"]},
                paddle.to_tensor([[[383, 253]], [[243, 253]]], dtype="int64"),
                paddle.to_tensor([[1], [1]], dtype="int64"),
                paddle.to_tensor([[[1, 1]], [[1, 1]]], dtype="int64"),
            ),
            (
                {0: ["差评", "不喜欢"], 1: ["好评", "不错"]},
                [0, 1],
                {0: ["差评", "不喜欢"], 1: ["好评", "不错"]},
                paddle.to_tensor([[[859, 480, 0], [16, 692, 811]], [[170, 480, 0], [16, 990, 0]]], dtype="int64"),
                paddle.to_tensor([[1, 1], [1, 1]], dtype="int64"),
                paddle.to_tensor([[[1, 1, 0], [1, 1, 1]], [[1, 1, 0], [1, 1, 0]]], dtype="int64"),
            ),
            (
                {1: ["很满意", "非常推荐"], 0: "避雷"},
                [0, 1],
                {0: ["避雷"], 1: ["很满意", "非常推荐"]},
                paddle.to_tensor(
                    [[[1166, 1048, 0, 0], [0, 0, 0, 0]], [[321, 596, 221, 0], [465, 223, 426, 1645]]], dtype="int64"
                ),
                paddle.to_tensor([[1, 0], [1, 1]], dtype="int64"),
                paddle.to_tensor([[[1, 1, 0, 0], [0, 0, 0, 0]], [[1, 1, 1, 0], [1, 1, 1, 1]]], dtype="int64"),
            ),
        ]
    )
    def test_initialization(
        self, label_words, labels, expected_words, expected_token_ids, expected_word_mask, expected_token_mask
    ):
        verb = ManualVerbalizer(label_words=label_words, tokenizer=self.tokenizer)
        self.assertEqual(verb.labels, labels)
        self.assertEqual(verb.label_words, expected_words)
        self.assertTrue(paddle.equal_all(verb.token_ids, expected_token_ids))
        self.assertTrue(paddle.equal_all(verb.word_mask, expected_word_mask))
        self.assertTrue(paddle.equal_all(verb.token_mask, expected_token_mask))

    def test_labels_property(self):
        verb = ManualVerbalizer(label_words=self.default_label_words, tokenizer=self.tokenizer)
        labels = ["差评", "好评"]
        label_words = {"差评": "避雷", "好评": "非常推荐"}
        expected_words = {"好评": ["非常推荐"], "差评": ["避雷"]}
        expected_token_ids = paddle.to_tensor([[[465, 223, 426, 1645]], [[1166, 1048, 0, 0]]], dtype="int64")
        expected_word_mask = paddle.to_tensor([[1], [1]], dtype="int64")
        expected_token_mask = paddle.to_tensor([[[1, 1, 1, 1]], [[1, 1, 0, 0]]], dtype="int64")
        with self.assertRaises(NotImplementedError):
            verb.labels = labels
        verb.label_words = label_words
        self.assertEqual(verb.labels, sorted(labels))
        self.assertEqual(verb.label_words, expected_words)
        self.assertTrue(paddle.equal_all(verb.token_ids, expected_token_ids))
        self.assertTrue(paddle.equal_all(verb.word_mask, expected_word_mask))
        self.assertTrue(paddle.equal_all(verb.token_mask, expected_token_mask))

    @parameterized.expand(
        [
            ("mean", [[0.5, 2.0], [-3.0, -1.5]]),
            ("max", [[1.0, 2.0], [-3.0, -1.0]]),
            ("first", [[0.0, 2.0], [-3.0, -1.0]]),
        ]
    )
    def test_aggregate_token(self, atype, expected_outputs):
        outputs = paddle.to_tensor([[[0, 1.0], [2, 3.0]], [[-3, -4.0], [-1, -2.0]]])
        token_mask = paddle.to_tensor([[[1, 1], [1, 0]], [[1, 0], [1, 1]]])
        outputs = self.default_verb.aggregate(outputs, token_mask, atype)
        self.assertEqual(outputs.tolist(), expected_outputs)

    @parameterized.expand(
        [
            ("mean", [0.5, 2.0]),
            ("max", [1.0, 2.0]),
            ("first", [0.0, 2.0]),
        ]
    )
    def test_aggregate_word(self, atype, expected_outputs):
        outputs = paddle.to_tensor([[0, 1.0], [2, 3.0]])
        word_mask = paddle.to_tensor([[1, 1], [1, 0]])
        outputs = self.default_verb.aggregate(outputs, word_mask, atype)
        self.assertEqual(outputs.tolist(), expected_outputs)

    @parameterized.expand(
        [
            ("mean", [[1, 2.0], [-2, -3.0]]),
            ("max", [[2, 3.0], [-1, -2.0]]),
            ("first", [[0, 1.0], [-3, -4.0]]),
            ("product", [[0, 3.0], [3, 8.0]]),
            ("invalid", None),
        ]
    )
    def test_aggregate_multiple_mask(self, atype, expected_outputs):
        outputs = paddle.to_tensor([[[0, 1.0], [2, 3.0]], [[-3, -4.0], [-1, -2.0]]])
        if atype == "invalid":
            with self.assertRaises(ValueError):
                self.default_verb.aggregate_multiple_mask(outputs, atype)
        else:
            outputs = self.default_verb.aggregate_multiple_mask(outputs, atype)
            self.assertEqual(outputs.tolist(), expected_outputs)

    def test_project(self):
        outputs = self.default_verb.project(paddle.rand([2, 1, 400]))
        self.assertEqual(outputs.shape, [2, 1, 2, 1])

    def test_process_outputs(self):
        outputs = paddle.rand([3, 2, 500])
        masked_positions = paddle.to_tensor([0, 2, 5])
        outputs = self.default_verb.process_outputs(outputs, masked_positions)
        self.assertEqual(outputs.shape, [3, 2])

    def test_normalize(self):
        outputs = paddle.rand([2, 1, 2, 3])
        self.assertAlmostEqual(self.default_verb.normalize(outputs)[0].sum().tolist()[0], 1, 6)
        self.assertAlmostEqual(self.default_verb.normalize(outputs)[1].sum().tolist()[0], 1, 6)

    def test_save_and_load(self):
        self.default_verb.save(save_path=self.save_path)
        verb = ManualVerbalizer.load_from(self.save_path, tokenizer=self.tokenizer)
        self.assertEqual(verb.label_words, self.default_verb.label_words)
        os.remove(os.path.join(self.save_path, "verbalizer_config.json"))
        os.rmdir(self.save_path)

    def test_encode_and_decode(self):
        label_words = {"负向": "不喜欢", "正向": "非常推荐"}
        verb = ManualVerbalizer(label_words=label_words, tokenizer=self.tokenizer)
        self.assertEqual(verb.convert_ids_to_labels(1), "负向")
        self.assertEqual(verb.convert_labels_to_ids("负向"), 1)


class SoftVerbalizerTest(unittest.TestCase):
    """
    Unittest for SoftVerbalizer
    """

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("__internal_testing__/tiny-random-ernie")
        cls.model = AutoModelForMaskedLM.from_pretrained("__internal_testing__/tiny-random-ernie")
        cls.default_label_words = {"正向": "很", "负向": "不"}
        cls.kwargs = {
            "token_aggregate_type": "first",
            "word_aggregate_type": "first",
            "mask_aggregate_type": "first",
            "post_log_softmax": False,
        }

    def test_kwargs(self):
        verb = SoftVerbalizer(
            label_words=self.default_label_words, tokenizer=self.tokenizer, model=self.model, **self.kwargs
        )
        self.assertEqual(verb.token_aggregate_type, "first")
        self.assertEqual(verb.word_aggregate_type, "first")
        self.assertEqual(verb.mask_aggregate_type, "first")
        self.assertFalse(verb.post_log_softmax)


if __name__ == "__main__":
    unittest.main()
