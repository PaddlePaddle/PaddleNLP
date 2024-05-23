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
from tempfile import TemporaryDirectory

import paddle
from parameterized import parameterized

from paddlenlp.prompt import (
    AutoTemplate,
    PromptModelForSequenceClassification,
    SoftVerbalizer,
)
from paddlenlp.taskflow import Taskflow
from paddlenlp.taskflow.text_classification import TextClassificationTask
from paddlenlp.transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class TestTextClassificationTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = TemporaryDirectory()
        tokenizer = AutoTokenizer.from_pretrained("__internal_testing__/ernie")

        # finetune (dynamic)
        cls.finetune_dygraph_model_path = os.path.join(cls.temp_dir.name, "finetune_dygraph")
        finetune_dygraph_model = AutoModelForSequenceClassification.from_pretrained(
            "__internal_testing__/ernie", num_classes=2
        )
        finetune_dygraph_model.save_pretrained(cls.finetune_dygraph_model_path)
        tokenizer.save_pretrained(cls.finetune_dygraph_model_path)

        # finetune (static)
        cls.finetune_static_model_path = os.path.join(cls.temp_dir.name, "finetune_static")
        input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
        ]
        finetune_static_model = paddle.jit.to_static(finetune_dygraph_model, input_spec=input_spec)
        paddle.jit.save(finetune_static_model, os.path.join(cls.finetune_static_model_path, "model"))
        tokenizer.save_pretrained(cls.finetune_static_model_path)

        # prompt (dynamic)
        cls.prompt_dygraph_model_path = os.path.join(cls.temp_dir.name, "prompt_dygraph")
        prompt_plm_model = AutoModelForMaskedLM.from_pretrained("__internal_testing__/ernie", num_classes=2)
        template = AutoTemplate.create_from("测试：", tokenizer, 16, model=prompt_plm_model)
        label_words = {"negative": ["负面"], "positive": ["正面"]}
        verbalizer = SoftVerbalizer(label_words, tokenizer, prompt_plm_model)
        prompt_dygraph_model = PromptModelForSequenceClassification(prompt_plm_model, template, verbalizer)

        template.save(cls.prompt_dygraph_model_path)
        verbalizer.save(cls.prompt_dygraph_model_path)
        tokenizer.save_pretrained(cls.prompt_dygraph_model_path)
        prompt_plm_model.save_pretrained(os.path.join(cls.prompt_dygraph_model_path, "plm"))
        state_dict = prompt_dygraph_model.state_dict()
        paddle.save(state_dict, os.path.join(cls.prompt_dygraph_model_path, "model_state.pdparams"))

        # prompt (static)
        cls.prompt_static_model_path = os.path.join(cls.temp_dir.name, "prompt_static")
        input_spec = prompt_dygraph_model.get_input_spec()
        prompt_static_model = paddle.jit.to_static(prompt_dygraph_model, input_spec=input_spec)

        template.save(cls.prompt_static_model_path)
        verbalizer.save(cls.prompt_static_model_path)
        tokenizer.save_pretrained(cls.prompt_static_model_path)
        prompt_plm_model.save_pretrained(os.path.join(cls.prompt_static_model_path, "plm"))
        paddle.jit.save(prompt_static_model, os.path.join(cls.prompt_static_model_path, "model"))

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    @parameterized.expand(
        [
            (1, "multi_class", "finetune"),
            # (1, "multi_class", "prompt"),  # TODO (paddle 2.5.1 breaks this test)
            (1, "multi_label", "finetune"),
            # (1, "multi_label", "prompt"),  # TODO (paddle 2.5.1 breaks this test)
        ]
    )
    def test_classification_task(self, batch_size, problem_type, model):
        # input_text is a tuple to simulate the args passed from Taskflow to TextClassificationTask
        input_text = (["百度", "深度学习框架", "飞桨", "PaddleNLP"],)
        id2label = {
            0: "negative",
            1: "positive",
        }
        if model == "finetune":
            dygraph_model_path = self.finetune_dygraph_model_path
            static_model_path = self.finetune_static_model_path
        else:
            dygraph_model_path = self.prompt_dygraph_model_path
            static_model_path = self.prompt_static_model_path

        dygraph_taskflow = TextClassificationTask(
            model=model,
            task="text_classification",
            task_path=dygraph_model_path,
            id2label=id2label,
            batch_size=batch_size,
            device_id=0,
            problem_type=problem_type,
        )

        dygraph_results = dygraph_taskflow(input_text)

        self.assertEqual(len(dygraph_results), len(input_text[0]))

        static_taskflow = TextClassificationTask(
            model=model,
            task="text_classification",
            is_static_model=True,
            task_path=static_model_path,
            id2label=id2label,
            batch_size=batch_size,
            device_id=0,
            problem_type=problem_type,
        )

        static_results = static_taskflow(input_text)
        self.assertEqual(len(static_results), len(input_text[0]))

        for dygraph_result, static_result in zip(dygraph_results, static_results):
            for dygraph_pred, static_pred in zip(dygraph_result["predictions"], static_result["predictions"]):
                self.assertEqual(dygraph_pred["label"], static_pred["label"])
                self.assertAlmostEqual(dygraph_pred["score"], static_pred["score"], delta=1e-6)
                # if multi_label, all predictions should be greater than the threshold
                if model == "multi_label":
                    self.assertGreater(dygraph_pred["score"], dygraph_taskflow.multilabel_threshold)

    @unittest.skip("numerical error")
    @parameterized.expand(
        [
            (1, "multi_class", "finetune"),
            (1, "multi_class", "prompt"),
            (1, "multi_label", "finetune"),
            (1, "multi_label", "prompt"),
        ]
    )
    def test_taskflow_task(self, batch_size, problem_type, mode):
        input_text = ["百度", "深度学习框架", "飞桨", "PaddleNLP"]
        id2label = {
            0: "negative",
            1: "positive",
        }
        if mode == "finetune":
            dygraph_model_path = self.finetune_dygraph_model_path
            static_model_path = self.finetune_static_model_path
        else:
            dygraph_model_path = self.prompt_dygraph_model_path
            static_model_path = self.prompt_static_model_path

        dygraph_taskflow = Taskflow(
            mode=mode,
            task="text_classification",
            task_path=dygraph_model_path,
            id2label=id2label,
            batch_size=batch_size,
            device_id=0,
            problem_type=problem_type,
        )

        dygraph_results = dygraph_taskflow(input_text)

        self.assertEqual(len(dygraph_results), len(input_text))

        static_taskflow = Taskflow(
            mode=mode,
            task="text_classification",
            is_static_model=True,
            task_path=static_model_path,
            id2label=id2label,
            batch_size=batch_size,
            device_id=0,
            problem_type=problem_type,
        )

        static_results = static_taskflow(input_text)
        self.assertEqual(len(static_results), len(input_text))

        for dygraph_result, static_result in zip(dygraph_results, static_results):
            for dygraph_pred, static_pred in zip(dygraph_result["predictions"], static_result["predictions"]):
                self.assertEqual(dygraph_pred["label"], static_pred["label"])
                self.assertAlmostEqual(dygraph_pred["score"], static_pred["score"], delta=1e-6)
                # if multi_label, all predictions should be greater than the threshold
                if mode == "multi_label":
                    self.assertGreater(dygraph_pred["score"], dygraph_taskflow.task_instance.multilabel_threshold)
