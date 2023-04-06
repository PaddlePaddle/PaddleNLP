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

import copy
import tempfile
import unittest

import paddle
from parameterized import parameterized

from paddlenlp.prompt import ManualVerbalizer, MaskedLMVerbalizer, SoftVerbalizer
from paddlenlp.prompt.verbalizer import MaskedLMIdentity
from paddlenlp.transformers import AutoModelForMaskedLM, AutoTokenizer
from paddlenlp.transformers.albert.modeling import AlbertMLMHead
from paddlenlp.transformers.ernie.modeling import ErnieLMPredictionHead


class VerbalizerTest(unittest.TestCase):
    """
    Unittest for Verbalizer
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
        cls.default_verb = ManualVerbalizer(label_words=cls.default_label_words, tokenizer=cls.tokenizer)

    @parameterized.expand(
        [
            (ManualVerbalizer,),
            (SoftVerbalizer,),
            (MaskedLMVerbalizer,),
        ]
    )
    def test_kwargs(self, class_name):
        model = copy.deepcopy(self.model)
        verb = class_name(tokenizer=self.tokenizer, label_words=self.default_label_words, model=model, **self.kwargs)
        self.assertEqual(verb.token_aggregate_type, "first")
        self.assertEqual(verb.word_aggregate_type, "first")
        self.assertEqual(verb.mask_aggregate_type, "first")
        self.assertFalse(verb.post_log_softmax)

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

    def test_project(self):
        outputs = self.default_verb.project(paddle.rand([2, 1, 400]))
        self.assertEqual(outputs.shape, [2, 1, 2, 1])

    def test_normalize(self):
        outputs = paddle.rand([2, 1, 2, 3])
        self.assertAlmostEqual(self.default_verb.normalize(outputs)[0].sum().tolist()[0], 1, 6)
        self.assertAlmostEqual(self.default_verb.normalize(outputs)[1].sum().tolist()[0], 1, 6)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.default_verb.save(save_path=tmpdirname)
            verb = ManualVerbalizer.load_from(tmpdirname, tokenizer=self.tokenizer)
            self.assertEqual(verb.label_words, self.default_verb.label_words)

    def test_encode_and_decode(self):
        label_words = {"负向": "不喜欢", "正向": "非常推荐"}
        verb = ManualVerbalizer(label_words=label_words, tokenizer=self.tokenizer)
        self.assertEqual(verb.convert_ids_to_labels(1), "负向")
        self.assertEqual(verb.convert_labels_to_ids("负向"), 1)

    @parameterized.expand(
        [
            (
                {"0": "负向", "1": "正向"},
                ["0", "1"],
                {"0": ["负向"], "1": ["正向"]},
                [[[383, 253]], [[243, 253]]],
                [[1], [1]],
                [[[1, 1]], [[1, 1]]],
            ),
            (
                {0: ["差评", "不喜欢"], 1: ["好评", "不错"]},
                [0, 1],
                {0: ["差评", "不喜欢"], 1: ["好评", "不错"]},
                [[[859, 480, 0], [16, 692, 811]], [[170, 480, 0], [16, 990, 0]]],
                [[1, 1], [1, 1]],
                [[[1, 1, 0], [1, 1, 1]], [[1, 1, 0], [1, 1, 0]]],
            ),
            (
                {1: ["很满意", "非常推荐"], 0: "避雷"},
                [0, 1],
                {0: ["避雷"], 1: ["很满意", "非常推荐"]},
                [[[1166, 1048, 0, 0], [0, 0, 0, 0]], [[321, 596, 221, 0], [465, 223, 426, 1645]]],
                [[1, 0], [1, 1]],
                [[[1, 1, 0, 0], [0, 0, 0, 0]], [[1, 1, 1, 0], [1, 1, 1, 1]]],
            ),
        ]
    )
    def test_manual_initialization(
        self, label_words, labels, expected_words, expected_token_ids, expected_word_mask, expected_token_mask
    ):
        verb = ManualVerbalizer(label_words=label_words, tokenizer=self.tokenizer)
        self.assertEqual(verb.labels, labels)
        self.assertEqual(verb.label_words, expected_words)
        self.assertTrue(paddle.equal_all(verb.token_ids, paddle.to_tensor(expected_token_ids, dtype="int64")))
        self.assertTrue(paddle.equal_all(verb.word_mask, paddle.to_tensor(expected_word_mask, dtype="int64")))
        self.assertTrue(paddle.equal_all(verb.token_mask, paddle.to_tensor(expected_token_mask, dtype="int64")))

    @parameterized.expand(
        [
            ("mean", [[1, 2.0], [-2, -3.0]]),
            ("max", [[2, 3.0], [-1, -2.0]]),
            ("first", [[0, 1.0], [-3, -4.0]]),
            ("product", [[0, 3.0], [3, 8.0]]),
            ("invalid", None),
        ]
    )
    def test_manual_aggregate_multiple_mask(self, atype, expected_outputs):
        outputs = paddle.to_tensor([[[0, 1.0], [2, 3.0]], [[-3, -4.0], [-1, -2.0]]])
        if atype == "invalid":
            with self.assertRaises(ValueError):
                self.default_verb.aggregate_multiple_mask(outputs, atype)
        else:
            outputs = self.default_verb.aggregate_multiple_mask(outputs, atype)
            self.assertEqual(outputs.tolist(), expected_outputs)

    def test_manual_process_outputs(self):
        outputs = paddle.rand([3, 2, 500])
        masked_positions = paddle.to_tensor([0, 2, 5])
        outputs = self.default_verb.process_outputs(outputs, masked_positions)
        self.assertEqual(outputs.shape, [3, 2])

    @parameterized.expand(
        [
            (
                "__internal_testing__/tiny-random-ernie",
                ["cls", "predictions", "decoder_weight"],
                ErnieLMPredictionHead,
                ["decoder_weight", "decoder_bias"],
                ["transform.weight", "transform.bias", "layer_norm.weight", "layer_norm.bias"],
            )
            # ,
            # (
            #     "albert-chinese-tiny",
            #     ["predictions", "decoder"],
            #     AlbertMLMHead,
            #     ["decoder.weight"],
            #     ["bias", "layer_norm.weight", "layer_norm.bias", "dense.weight", "dense.bias"],
            # ),
        ]
    )
    def test_soft_initialization(self, model_name, head_name, head_class, head_params, non_head_params):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        verb = SoftVerbalizer(self.default_label_words, tokenizer, model)

        self.assertEqual(verb.head_name, head_name)
        self.assertTrue(isinstance(verb.head, head_class))
        self.assertTrue(isinstance(getattr(model, head_name[0]), MaskedLMIdentity))
        module = getattr(verb.head, verb.head_name[-1])
        module = module.weight if isinstance(module, paddle.nn.Linear) else module
        self.assertTrue(len(self.default_label_words) in module.shape)

        self.assertEqual([x[0] for x in verb.head_parameters()], head_params)
        self.assertEqual([x[0] for x in verb.non_head_parameters()], non_head_params)

    def test_soft_process_outputs(self):
        verb = SoftVerbalizer(self.default_label_words, self.tokenizer, copy.deepcopy(self.model))
        outputs = paddle.rand([3, 2, 8])
        masked_positions = paddle.to_tensor([0, 2, 5])
        outputs = verb.process_outputs(outputs, masked_positions)

    @parameterized.expand(
        [
            (
                {"0": "负向", "1": "正向"},
                {"0": ["负向"], "1": ["正向"]},
            ),
            (
                {1: ["好评", "不错"], 0: ["差评", "不喜欢"]},
                {0: ["差评"], 1: ["好评"]},
            ),
            (
                {1: "很满意", 0: "避雷"},
                None,
            ),
        ]
    )
    def test_maskedlm_initialization(self, label_words, expected_label_words):
        if expected_label_words is None:
            with self.assertRaises(ValueError):
                verb = MaskedLMVerbalizer(label_words, self.tokenizer)
        else:
            verb = MaskedLMVerbalizer(label_words, self.tokenizer)
            self.assertEqual(verb.label_words, expected_label_words)

    @parameterized.expand([("mean",), ("max",), ("first",), ("product",), ("sum",), ("invalid",)])
    def test_maskedlm_aggregate_multiple_mask(self, atype):
        label_words = {"0": "负向", "1": "正向"}
        verb = MaskedLMVerbalizer(label_words, self.tokenizer)
        outputs = paddle.rand([3, 2, 500])
        if atype == "invalid":
            with self.assertRaises(ValueError):
                verb.aggregate_multiple_mask(outputs, atype)
        else:
            outputs = verb.aggregate_multiple_mask(outputs, atype)
            self.assertEqual(outputs.shape, [3, 2])


if __name__ == "__main__":
    unittest.main()
