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

import unittest

from paddlenlp.prompt import (
    AutoTemplate,
    PromptDataCollatorWithPadding,
    PromptModelForSequenceClassification,
    SoftVerbalizer,
)
from paddlenlp.transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class PromptModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("__internal_testing__/tiny-random-ernie")
        cls.model = AutoModelForMaskedLM.from_pretrained("__internal_testing__/tiny-random-ernie")
        cls.num_labels = 2
        cls.seq_cls_model = AutoModelForSequenceClassification.from_pretrained(
            "__internal_testing__/tiny-random-ernie", num_labels=cls.num_labels
        )

        cls.template = AutoTemplate.create_from(
            prompt="{'soft'}{'text': 'text'}{'mask'}", tokenizer=cls.tokenizer, max_length=512, model=cls.model
        )
        cls.label_words = {0: "0", 1: "1", 2: "2"}
        cls.verbalizer = SoftVerbalizer(cls.label_words, cls.tokenizer, cls.model)
        cls.data_collator = PromptDataCollatorWithPadding(cls.tokenizer, padding=True, return_tensors="pd")
        cls.prompt_model = PromptModelForSequenceClassification(cls.model, cls.template, cls.verbalizer)

    def test_sequence_classification_no_labels(self):
        examples = [{"text": "百度飞桨深度学习框架"}, {"text": "这是一个测试"}]
        encoded_examples = [self.template(i) for i in examples]
        logits, hidden_states = self.prompt_model(**self.data_collator(encoded_examples), return_hidden_states=True)
        self.assertEqual(logits.shape[0], len(examples))
        self.assertEqual(logits.shape[1], len(self.label_words))
        self.assertEqual(hidden_states.shape[0], len(examples))

        model_outputs = self.prompt_model(
            **self.data_collator(encoded_examples), return_dict=True, return_hidden_states=True
        )
        self.assertIsNone(model_outputs.loss)
        self.assertEqual(model_outputs.logits.shape[0], len(examples))
        self.assertEqual(model_outputs.logits.shape[1], len(self.label_words))
        self.assertEqual(model_outputs.hidden_states.shape[0], len(examples))

    def test_sequence_classification_with_labels(self):
        examples = [{"text": "百度飞桨深度学习框架", "labels": 0}, {"text": "这是一个测试", "labels": 1}]
        encoded_examples = [self.template(i) for i in examples]
        loss, logits, hidden_states = self.prompt_model(
            **self.data_collator(encoded_examples), return_hidden_states=True
        )
        self.assertIsNotNone(loss)
        self.assertEqual(logits.shape[0], len(examples))
        self.assertEqual(logits.shape[1], len(self.label_words))
        self.assertEqual(hidden_states.shape[0], len(examples))

        model_outputs = self.prompt_model(
            **self.data_collator(encoded_examples), return_dict=True, return_hidden_states=True
        )
        self.assertIsNotNone(model_outputs.loss)
        self.assertEqual(model_outputs.logits.shape[0], len(examples))
        self.assertEqual(model_outputs.logits.shape[1], len(self.label_words))
        self.assertEqual(model_outputs.hidden_states.shape[0], len(examples))

    def test_efl_no_labels(self):
        prompt_model = PromptModelForSequenceClassification(self.seq_cls_model, self.template, verbalizer=None)
        examples = [{"text": "百度飞桨深度学习框架"}, {"text": "这是一个测试"}]
        encoded_examples = [self.template(i) for i in examples]
        logits, hidden_states = prompt_model(**self.data_collator(encoded_examples), return_hidden_states=True)
        self.assertEqual(logits.shape[0], len(examples))
        self.assertEqual(logits.shape[1], self.num_labels)
        self.assertEqual(hidden_states.shape[0], len(examples))

        model_outputs = prompt_model(
            **self.data_collator(encoded_examples), return_dict=True, return_hidden_states=True
        )
        self.assertIsNone(model_outputs.loss)
        self.assertEqual(model_outputs.logits.shape[0], len(examples))
        self.assertEqual(model_outputs.logits.shape[1], self.num_labels)
        self.assertEqual(model_outputs.hidden_states.shape[0], len(examples))

    def test_efl_with_labels(self):
        prompt_model = PromptModelForSequenceClassification(self.seq_cls_model, self.template, verbalizer=None)
        examples = [{"text": "百度飞桨深度学习框架", "labels": 0}, {"text": "这是一个测试", "labels": 1}]
        encoded_examples = [self.template(i) for i in examples]
        loss, logits, hidden_states = prompt_model(**self.data_collator(encoded_examples), return_hidden_states=True)
        self.assertIsNotNone(loss)
        self.assertEqual(logits.shape[0], len(examples))
        self.assertEqual(logits.shape[1], self.num_labels)
        self.assertEqual(hidden_states.shape[0], len(examples))

        model_outputs = prompt_model(
            **self.data_collator(encoded_examples), return_dict=True, return_hidden_states=True
        )
        self.assertIsNotNone(model_outputs.loss)
        self.assertEqual(model_outputs.logits.shape[0], len(examples))
        self.assertEqual(model_outputs.logits.shape[1], self.num_labels)
        self.assertEqual(model_outputs.hidden_states.shape[0], len(examples))


if __name__ == "__main__":
    unittest.main()
