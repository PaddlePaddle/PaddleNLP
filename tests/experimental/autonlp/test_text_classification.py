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
import os
import unittest
from tempfile import TemporaryDirectory

import ray
from hyperopt import hp
from pandas import DataFrame
from parameterized import parameterized

from paddlenlp.datasets import load_dataset
from paddlenlp.experimental.autonlp import AutoTrainerForTextClassification
from tests.testing_utils import get_tests_dir, slow

finetune_model_candidate = {
    "trainer_type": "Trainer",
    "TrainingArguments.max_steps": 5,
    "TrainingArguments.per_device_train_batch_size": 2,
    "TrainingArguments.per_device_eval_batch_size": 2,
    "TrainingArguments.model_name_or_path": hp.choice("finetune_models", ["__internal_testing__/tiny-random-bert"]),
}
prompt_model_candidate = {
    "trainer_type": "PromptTrainer",
    "template.prompt": "“{'text': 'sentence'}”这句话是关于{'mask'}的",
    "PromptTuningArguments.max_steps": 5,
    "PromptTuningArguments.per_device_train_batch_size": 2,
    "PromptTuningArguments.per_device_eval_batch_size": 2,
    "PromptTuningArguments.model_name_or_path": hp.choice("prompt_models", ["__internal_testing__/tiny-random-bert"]),
}


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
    @classmethod
    def setUpClass(cls):
        fixture_path = get_tests_dir(os.path.join("fixtures", "dummy"))
        cls.multi_class_train_ds, cls.multi_class_dev_ds = load_dataset(
            "clue",
            "tnews",
            data_files=[
                os.path.join(fixture_path, "tnews", "train.json"),
                os.path.join(fixture_path, "tnews", "dev.json"),
            ],
            lazy=False,
        )
        cls.multi_label_train_ds = load_dataset(
            read_multi_label_dataset, path=os.path.join(fixture_path, "divorce", "train.txt"), lazy=False
        )
        cls.multi_label_dev_ds = load_dataset(
            read_multi_label_dataset,
            path=os.path.join(fixture_path, "divorce", "dev.txt"),
            lazy=False,
        )
        ray.init(local_mode=True)

    @parameterized.expand(
        [
            ([finetune_model_candidate], {"TrainingArguments.max_steps": 2}),
            ([finetune_model_candidate], None),
            ([prompt_model_candidate], None),
            ([finetune_model_candidate, prompt_model_candidate], None),
        ]
    )
    def test_multiclass(self, custom_model_candidate, hp_overrides):
        with TemporaryDirectory() as temp_dir_path:
            train_ds = copy.deepcopy(self.multi_class_train_ds)
            dev_ds = copy.deepcopy(self.multi_class_dev_ds)
            num_models = 1
            # create auto trainer and train
            auto_trainer = AutoTrainerForTextClassification(
                train_dataset=train_ds,
                eval_dataset=dev_ds,
                label_column="label_desc",
                text_column="sentence",
                language="Chinese",
                output_dir=temp_dir_path,
                problem_type="multi_class",
            )
            auto_trainer.train(
                num_cpus=1,
                num_gpus=0,
                max_concurrent_trials=1,
                num_models=num_models,
                custom_model_candidates=custom_model_candidate,
                hp_overrides=hp_overrides,
            )

            # check is training is valid
            self.assertEqual(len(auto_trainer.training_results.errors), 0)
            self.assertEqual(len(auto_trainer.training_results), num_models)

            # test show_training_results
            results_df = auto_trainer.show_training_results()
            self.assertIsInstance(results_df, DataFrame)
            self.assertEqual(len(results_df), num_models)

            # test hp override
            if hp_overrides is not None:
                for hp_key, hp_value in hp_overrides.items():
                    result_hp_key = f"config/candidates/{hp_key}"
                    self.assertEqual(results_df[result_hp_key][0], hp_value)

            # test save
            save_path = os.path.join(auto_trainer._get_model_result().log_dir, auto_trainer.save_path)
            self.assertTrue(os.path.exists(os.path.join(save_path, "model_state.pdparams")))
            self.assertTrue(os.path.exists(os.path.join(save_path, "tokenizer_config.json")))
            if len(custom_model_candidate) == 1 and custom_model_candidate[0]["trainer_type"] == "PromptTrainer":
                self.assertTrue(os.path.exists(os.path.join(save_path, "template_config.json")))
                self.assertTrue(os.path.exists(os.path.join(save_path, "verbalizer_config.json")))

            # test export
            temp_export_path = os.path.join(temp_dir_path, "test_export")
            auto_trainer.export(export_path=temp_export_path)
            self.assertTrue(os.path.exists(os.path.join(temp_export_path, "model.pdmodel")))
            self.assertTrue(os.path.exists(os.path.join(temp_export_path, "taskflow_config.json")))
            self.assertTrue(os.path.exists(os.path.join(temp_export_path, "tokenizer_config.json")))

            if len(custom_model_candidate) == 1 and custom_model_candidate[0]["trainer_type"] == "PromptTrainer":
                self.assertTrue(os.path.exists(os.path.join(temp_export_path, "template_config.json")))

            # test invalid export
            temp_export_path = os.path.join(temp_dir_path, "invalid_export")
            with self.assertRaises(LookupError):
                auto_trainer.export(export_path=temp_export_path, trial_id="invalid_trial_id")

            # test evaluate
            copy_dev_ds = copy.deepcopy(self.multi_class_dev_ds)
            eval_metrics1 = auto_trainer.evaluate()
            eval_metrics2 = auto_trainer.evaluate(eval_dataset=copy_dev_ds)
            self.assertEqual(
                eval_metrics1[auto_trainer.metric_for_best_model],
                eval_metrics2[auto_trainer.metric_for_best_model],
            )

            # test taskflow
            taskflow = auto_trainer.to_taskflow(export_path=temp_export_path)
            test_inputs = [dev_ds[0]["sentence"], dev_ds[1]["sentence"]]
            test_results = taskflow(test_inputs)
            self.assertEqual(len(test_results), len(test_inputs))
            for test_result in test_results:
                for prediction in test_result["predictions"]:
                    self.assertIn(prediction["label"], auto_trainer.label2id)

            # test training_path
            self.assertFalse(os.path.exists(os.path.join(auto_trainer.training_path)))

    @parameterized.expand(
        [
            ([finetune_model_candidate], {"TrainingArguments.max_steps": 2}),
            ([finetune_model_candidate], None),
            ([prompt_model_candidate], None),
            ([finetune_model_candidate, prompt_model_candidate], None),
        ]
    )
    def test_multilabel(self, custom_model_candidate, hp_overrides):
        with TemporaryDirectory() as temp_dir_path:
            train_ds = copy.deepcopy(self.multi_label_train_ds)
            dev_ds = copy.deepcopy(self.multi_label_dev_ds)
            num_models = 1
            # create auto trainer and train
            auto_trainer = AutoTrainerForTextClassification(
                train_dataset=train_ds,
                eval_dataset=dev_ds,
                label_column="labels",
                text_column="sentence",
                language="Chinese",
                output_dir=temp_dir_path,
                problem_type="multi_label",
            )
            auto_trainer.train(
                num_cpus=1,
                num_gpus=0,
                max_concurrent_trials=1,
                num_models=num_models,
                custom_model_candidates=custom_model_candidate,
                hp_overrides=hp_overrides,
            )

            # check is training is valid
            self.assertEqual(len(auto_trainer.training_results.errors), 0)
            self.assertEqual(len(auto_trainer.training_results), num_models)

            # test show_training_results
            results_df = auto_trainer.show_training_results()
            self.assertIsInstance(results_df, DataFrame)
            self.assertEqual(len(results_df), num_models)

            # test hp override
            if hp_overrides is not None:
                for hp_key, hp_value in hp_overrides.items():
                    result_hp_key = f"config/candidates/{hp_key}"
                    self.assertEqual(results_df[result_hp_key][0], hp_value)

            # test save
            save_path = os.path.join(auto_trainer._get_model_result().log_dir, auto_trainer.save_path)
            self.assertTrue(os.path.exists(os.path.join(save_path, "model_state.pdparams")))
            self.assertTrue(os.path.exists(os.path.join(save_path, "tokenizer_config.json")))
            if len(custom_model_candidate) == 1 and custom_model_candidate[0]["trainer_type"] == "PromptTrainer":
                self.assertTrue(os.path.exists(os.path.join(save_path, "template_config.json")))
                self.assertTrue(os.path.exists(os.path.join(save_path, "verbalizer_config.json")))

            # test export
            temp_export_path = os.path.join(temp_dir_path, "test_export")
            auto_trainer.export(export_path=temp_export_path)
            self.assertTrue(os.path.exists(os.path.join(temp_export_path, "model.pdmodel")))
            self.assertTrue(os.path.exists(os.path.join(temp_export_path, "taskflow_config.json")))
            self.assertTrue(os.path.exists(os.path.join(temp_export_path, "tokenizer_config.json")))

            if len(custom_model_candidate) == 1 and custom_model_candidate[0]["trainer_type"] == "PromptTrainer":
                self.assertTrue(os.path.exists(os.path.join(temp_export_path, "template_config.json")))
                self.assertTrue(os.path.exists(os.path.join(temp_export_path, "verbalizer_config.json")))

            # test evaluate
            copy_dev_ds = copy.deepcopy(self.multi_label_dev_ds)
            eval_metrics1 = auto_trainer.evaluate()
            eval_metrics2 = auto_trainer.evaluate(eval_dataset=copy_dev_ds)
            self.assertEqual(
                eval_metrics1[auto_trainer.metric_for_best_model],
                eval_metrics2[auto_trainer.metric_for_best_model],
            )

            # test invalid export
            temp_export_path = os.path.join(temp_dir_path, "invalid_export")
            with self.assertRaises(LookupError):
                auto_trainer.export(export_path=temp_export_path, trial_id="invalid_trial_id")

            # test taskflow
            taskflow = auto_trainer.to_taskflow(export_path=temp_export_path)
            test_inputs = [dev_ds[0]["sentence"], dev_ds[1]["sentence"]]
            test_results = taskflow(test_inputs)
            self.assertEqual(len(test_results), len(test_inputs))
            for test_result in test_results:
                for prediction in test_result["predictions"]:
                    self.assertIn(prediction["label"], auto_trainer.label2id)
                    self.assertGreater(prediction["score"], taskflow.task_instance.multilabel_threshold)

            # test training_path
            self.assertFalse(os.path.exists(os.path.join(auto_trainer.training_path)))

    @parameterized.expand(
        [
            (
                "Chinese",
                {
                    "TrainingArguments.max_steps": 2,
                    "TrainingArguments.per_device_train_batch_size": 1,
                    "TrainingArguments.per_device_eval_batch_size": 1,
                },
            ),
            (
                "English",
                {
                    "TrainingArguments.max_steps": 2,
                    "TrainingArguments.per_device_train_batch_size": 1,
                    "TrainingArguments.per_device_eval_batch_size": 1,
                },
            ),
        ]
    )
    @slow
    def test_default_model_candidate(self, language, hp_overrides):
        with TemporaryDirectory() as temp_dir_path:
            train_ds = copy.deepcopy(self.multi_class_train_ds)
            dev_ds = copy.deepcopy(self.multi_class_dev_ds)
            num_models = 2
            # create auto trainer and train
            auto_trainer = AutoTrainerForTextClassification(
                train_dataset=train_ds,
                eval_dataset=dev_ds,
                label_column="label_desc",
                text_column="sentence",
                language=language,
                output_dir=temp_dir_path,
                problem_type="multi_class",
            )
            auto_trainer.train(
                num_cpus=0,
                num_gpus=1,
                max_concurrent_trials=1,
                num_models=num_models,
                hp_overrides=hp_overrides,
            )

            # check is training is valid
            self.assertEqual(len(auto_trainer.training_results.errors), 0)
            self.assertEqual(len(auto_trainer.training_results), num_models)

            # test show_training_results
            results_df = auto_trainer.show_training_results()
            self.assertIsInstance(results_df, DataFrame)
            self.assertEqual(len(results_df), num_models)

            # test hp override
            if hp_overrides is not None:
                for hp_key, hp_value in hp_overrides.items():
                    result_hp_key = f"config/candidates/{hp_key}"
                    self.assertEqual(results_df[result_hp_key][0], hp_value)

            # test save
            save_path = os.path.join(auto_trainer._get_model_result().log_dir, auto_trainer.save_path)
            self.assertTrue(os.path.exists(os.path.join(save_path, "model_state.pdparams")))
            self.assertTrue(os.path.exists(os.path.join(save_path, "tokenizer_config.json")))

            # test export
            temp_export_path = os.path.join(temp_dir_path, "test_export")
            auto_trainer.export(export_path=temp_export_path)
            self.assertTrue(os.path.exists(os.path.join(temp_export_path, "model.pdmodel")))
            self.assertTrue(os.path.exists(os.path.join(temp_export_path, "taskflow_config.json")))
            self.assertTrue(os.path.exists(os.path.join(temp_export_path, "tokenizer_config.json")))

            # test invalid export
            temp_export_path = os.path.join(temp_dir_path, "invalid_export")
            with self.assertRaises(LookupError):
                auto_trainer.export(export_path=temp_export_path, trial_id="invalid_trial_id")

            # test evaluate
            copy_dev_ds = copy.deepcopy(self.multi_class_dev_ds)
            eval_metrics1 = auto_trainer.evaluate()
            eval_metrics2 = auto_trainer.evaluate(eval_dataset=copy_dev_ds)
            self.assertEqual(
                eval_metrics1[auto_trainer.metric_for_best_model],
                eval_metrics2[auto_trainer.metric_for_best_model],
            )

            # test taskflow
            taskflow = auto_trainer.to_taskflow(export_path=temp_export_path)
            test_inputs = [dev_ds[0]["sentence"], dev_ds[1]["sentence"]]
            test_results = taskflow(test_inputs)
            self.assertEqual(len(test_results), len(test_inputs))
            for test_result in test_results:
                for prediction in test_result["predictions"]:
                    self.assertIn(prediction["label"], auto_trainer.label2id)

            # test training_path
            self.assertFalse(os.path.exists(os.path.join(auto_trainer.training_path)))

    def test_untrained_auto_trainer(self):
        with TemporaryDirectory() as temp_dir:
            train_ds = copy.deepcopy(self.multi_class_train_ds)
            dev_ds = copy.deepcopy(self.multi_class_dev_ds)
            auto_trainer = AutoTrainerForTextClassification(
                train_dataset=train_ds,
                eval_dataset=dev_ds,
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

    def test_unsupported_languages(self):
        with TemporaryDirectory() as temp_dir:
            train_ds = copy.deepcopy(self.multi_class_train_ds)
            dev_ds = copy.deepcopy(self.multi_class_dev_ds)
            with self.assertRaises(ValueError):
                AutoTrainerForTextClassification(
                    train_dataset=train_ds,
                    eval_dataset=dev_ds,
                    label_column="label_desc",
                    text_column="sentence",
                    language="Spanish",  # spanish is unsupported for now
                    output_dir=temp_dir,
                )

    def test_model_language_filter(self):
        with TemporaryDirectory() as temp_dir:
            train_ds = copy.deepcopy(self.multi_class_train_ds)
            dev_ds = copy.deepcopy(self.multi_class_dev_ds)
            auto_trainer = AutoTrainerForTextClassification(
                train_dataset=train_ds,
                eval_dataset=dev_ds,
                label_column="label_desc",
                text_column="sentence",
                language="Chinese",
                output_dir=temp_dir,
            )
            for language in auto_trainer.supported_languages:
                model_candidates = auto_trainer._filter_model_candidates(language=language)
                for candidate in model_candidates:
                    self.assertEqual(candidate["language"], language)

    def test_id2label_label_not_found(self):
        with TemporaryDirectory() as temp_dir:
            train_ds = copy.deepcopy(self.multi_class_train_ds)
            # multi class
            dev_ds = copy.deepcopy(self.multi_class_dev_ds)
            with self.assertRaises(ValueError):
                AutoTrainerForTextClassification(
                    train_dataset=train_ds,
                    eval_dataset=dev_ds,
                    label_column="label_desc",
                    text_column="sentence",
                    language="Chinese",
                    output_dir=temp_dir,
                    id2label={0: "negative", 1: "positive"},
                    problem_type="multi_class",
                )

            # multi label
            dev_ds = copy.deepcopy(self.multi_label_dev_ds)
            with self.assertRaises(ValueError):
                AutoTrainerForTextClassification(
                    train_dataset=train_ds,
                    eval_dataset=dev_ds,
                    label_column="label_desc",
                    text_column="sentence",
                    language="Chinese",
                    output_dir=temp_dir,
                    id2label={0: "negative", 1: "positive"},
                    problem_type="multi_label",
                )


if __name__ == "__main__":
    unittest.main()
