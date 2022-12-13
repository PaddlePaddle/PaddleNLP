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

from pandas import DataFrame

from paddlenlp.datasets import load_dataset
from paddlenlp.experimental.autonlp import AutoTrainerForTextClassification
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from tests.testing_utils import get_tests_dir


def read_multi_label_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items = line.strip().split("\t")
            if len(items) == 1:
                sentence = items[0]
                labels = []
            else:
                sentence = "".join(items[:-1])
                label = items[-1]
                labels = label.split(",")
            yield {"sentence": sentence, "labels": labels}


class TestAutoTrainerForTextClassification(unittest.TestCase):
    def test_multiclass_finetune(self):
        with TemporaryDirectory() as temp_dir_path:
            fixture_path = get_tests_dir(os.path.join("fixtures", "dummy"))
            train_ds, dev_ds = load_dataset(
                "clue",
                "tnews",
                data_files=[
                    os.path.join(fixture_path, "tnews", "train.json"),
                    os.path.join(fixture_path, "tnews", "dev.json"),
                ],
                lazy=False,
            )
            num_models = 2
            # create auto trainer and train
            auto_trainer = AutoTrainerForTextClassification(
                label_column="label_desc",
                text_column="sentence",
                language="Chinese",
                output_dir=temp_dir_path,
            )
            auto_trainer.train(
                train_ds,
                dev_ds,
                preset="finetune_test",
                num_cpus=1,
                num_gpus=0,
                max_concurrent_trials=1,
                num_models=num_models,
            )

            # check is training is valid
            self.assertEqual(len(auto_trainer.training_results.errors), 0)
            self.assertEqual(len(auto_trainer.training_results), num_models)

            # test show_training_results
            results_df = auto_trainer.show_training_results()
            self.assertIsInstance(results_df, DataFrame)
            self.assertEqual(len(results_df), num_models)

            # test export
            temp_export_path = os.path.join(temp_dir_path, "test_export")
            auto_trainer.export(export_path=temp_export_path)
            reloaded_model = AutoModelForSequenceClassification.from_pretrained(temp_export_path)
            reloaded_tokenizer = AutoTokenizer.from_pretrained(temp_export_path)
            input_features = reloaded_tokenizer(dev_ds[0]["sentence"], return_tensors="pd")
            model_outputs = reloaded_model(**input_features)
            self.assertEqual(model_outputs.shape, [1, len(auto_trainer.id2label)])

            # test invalid export
            temp_export_path = os.path.join(temp_dir_path, "invalid_export")
            with self.assertRaises(LookupError):
                auto_trainer.export(export_path=temp_export_path, trial_id="invalid_trial_id")

            # test taskflow
            taskflow = auto_trainer.to_taskflow()
            test_inputs = [dev_ds[0]["sentence"], dev_ds[1]["sentence"]]
            test_results = taskflow(test_inputs)
            self.assertEqual(len(test_results), len(test_inputs))
            for test_result in test_results:
                self.assertIn(test_result["label"], auto_trainer.label2id)

    def test_multiclass_prompt(self):
        with TemporaryDirectory() as temp_dir_path:
            fixture_path = get_tests_dir(os.path.join("fixtures", "dummy"))
            train_ds, dev_ds = load_dataset(
                "clue",
                "tnews",
                data_files=[
                    os.path.join(fixture_path, "tnews", "train.json"),
                    os.path.join(fixture_path, "tnews", "dev.json"),
                ],
                lazy=False,
            )
            num_models = 2
            # create auto trainer and train
            auto_trainer = AutoTrainerForTextClassification(
                label_column="label_desc",
                text_column="sentence",
                language="Chinese",
                output_dir=temp_dir_path,
            )
            auto_trainer.train(
                train_ds,
                dev_ds,
                preset="prompt_test",
                num_cpus=1,
                num_gpus=0,
                max_concurrent_trials=1,
                num_models=num_models,
            )

            # check is training is valid
            self.assertEqual(len(auto_trainer.training_results.errors), 0)
            self.assertEqual(len(auto_trainer.training_results), num_models)

            # test show_training_results
            results_df = auto_trainer.show_training_results()
            self.assertIsInstance(results_df, DataFrame)
            self.assertEqual(len(results_df), num_models)

            # test export
            temp_export_path = os.path.join(temp_dir_path, "test_export")
            auto_trainer.export(export_path=temp_export_path)
            reloaded_model = AutoModelForSequenceClassification.from_pretrained(temp_export_path)
            reloaded_tokenizer = AutoTokenizer.from_pretrained(temp_export_path)
            input_features = reloaded_tokenizer(dev_ds[0]["sentence"], return_tensors="pd")
            model_outputs = reloaded_model(**input_features)
            self.assertEqual(model_outputs.shape, [1, len(auto_trainer.id2label)])

    def test_multilabel_finetune(self):
        with TemporaryDirectory() as temp_dir_path:
            fixture_path = get_tests_dir(os.path.join("fixtures", "dummy"))
            train_ds = load_dataset(
                read_multi_label_dataset,
                path=os.path.join(fixture_path, "divorce", "train.txt"),
                lazy=False,
            )
            dev_ds = load_dataset(
                read_multi_label_dataset,
                path=os.path.join(fixture_path, "divorce", "dev.txt"),
                lazy=False,
            )
            num_models = 2
            # create auto trainer and train
            auto_trainer = AutoTrainerForTextClassification(
                label_column="labels",
                text_column="sentence",
                language="Chinese",
                output_dir=temp_dir_path,
                problem_type="multi_label",
            )
            auto_trainer.train(
                train_ds,
                dev_ds,
                preset="finetune_test",
                num_cpus=1,
                num_gpus=0,
                max_concurrent_trials=1,
                num_models=num_models,
            )

            # check is training is valid
            self.assertEqual(len(auto_trainer.training_results.errors), 0)
            self.assertEqual(len(auto_trainer.training_results), num_models)

            # test show_training_results
            results_df = auto_trainer.show_training_results()
            self.assertIsInstance(results_df, DataFrame)
            self.assertEqual(len(results_df), num_models)

            # test export
            temp_export_path = os.path.join(temp_dir_path, "test_export")
            auto_trainer.export(export_path=temp_export_path)
            reloaded_model = AutoModelForSequenceClassification.from_pretrained(temp_export_path)
            reloaded_tokenizer = AutoTokenizer.from_pretrained(temp_export_path)
            input_features = reloaded_tokenizer(dev_ds[0]["sentence"], return_tensors="pd")
            model_outputs = reloaded_model(**input_features)
            self.assertEqual(model_outputs.shape, [1, len(auto_trainer.id2label)])

            # test invalid export
            temp_export_path = os.path.join(temp_dir_path, "invalid_export")
            with self.assertRaises(LookupError):
                auto_trainer.export(export_path=temp_export_path, trial_id="invalid_trial_id")

            # test taskflow
            # TODO: add multi-label taskflow support

    def test_untrained_auto_trainer(self):
        with TemporaryDirectory() as temp_dir:
            auto_trainer = AutoTrainerForTextClassification(
                label_column="label_desc",
                text_column="sentence",
                language="Chinese",
                output_dir=temp_dir,
            )

            with self.assertRaises(AttributeError):
                # test show results
                auto_trainer.show_training_results()

                # test export
                auto_trainer.export(temp_dir)


if __name__ == "__main__":
    unittest.main()
