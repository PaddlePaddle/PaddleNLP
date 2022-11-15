# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import shutil, tempfile
import unittest
from paddlenlp.transformers import AutoTokenizer, AutoModelForSequenceClassification

from pandas import DataFrame
from paddlenlp.datasets import load_dataset
from paddlenlp.experimental.autonlp import AutoTrainerForTextClassification
from tests.testing_utils import get_tests_dir


class TestAutoTrainerForTextClassification(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.fixture_path = get_tests_dir(os.path.join("fixtures", "dummy"))

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.temp_dir)

    def test_multiclass_classification(self):
        train_ds, dev_ds = load_dataset(
            "clue",
            "tnews",
            data_files=[
                os.path.join(self.fixture_path, "tnews", "train.json"),
                os.path.join(self.fixture_path, "tnews", "dev.json"),
            ],
            lazy=False,
        )
        num_models = 2
        auto_trainer = AutoTrainerForTextClassification(
            label_column="label_desc",
            text_column="sentence",
            language="Chinese",
            output_dir=self.temp_dir,
        )
        auto_trainer.train(
            train_ds,
            dev_ds,
            preset="test",
            num_cpus=1,
            num_gpus=0,
            max_concurrent_trials=1,
            num_models=num_models,
        )

        # check is training is valid
        self.assertEqual(len(auto_trainer.training_results.errors), 0)
        self.assertEqual(len(auto_trainer.training_results), num_models)

        # test show results
        self.assertIsInstance(auto_trainer.show_training_results(), DataFrame)

        # test export
        temp_export_path = os.path.join(self.temp_dir, "test_export")
        auto_trainer.export(export_path=temp_export_path)
        reloaded_model = AutoModelForSequenceClassification.from_pretrained(
            temp_export_path
        )
        reloaded_tokenizer = AutoTokenizer.from_pretrained(temp_export_path)
        input_features = reloaded_tokenizer(dev_ds[0]["sentence"], return_tensors="pd")
        model_outputs = reloaded_model(**input_features)
        self.assertEqual(model_outputs.shape, [1, len(auto_trainer.id2label)])

        # test invalid export
        with self.assertRaises(LookupError):
            auto_trainer.export(export_path=temp_export_path, trial_id="invalid_trial")

    def test_multiclass_classification_exceptions(self):
        auto_trainer = AutoTrainerForTextClassification(
            label_column="label_desc",
            text_column="sentence",
            language="Chinese",
            output_dir=self.temp_dir,
        )

        with self.assertRaises(AttributeError):
            # test show results
            auto_trainer.show_training_results()

            # test export
            auto_trainer.export(self.temp_dir)


if __name__ == "__main__":
    unittest.main()
