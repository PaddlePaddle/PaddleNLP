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
import unittest

from paddlenlp.datasets import load_dataset
from paddlenlp.experimental.autonlp import AutoTrainerForTextClassification
from tests.testing_utils import get_tests_dir


class TestAutoTrainerForTextClassification(unittest.TestCase):

    def setUp(self):
        self.fixture_path = get_tests_dir(os.path.join("fixtures", "dummy"))

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
            label_column="label_desc", text_column="sentence")
        auto_trainer.train(
            train_ds,
            dev_ds,
            preset="test",
            num_cpus=1,
            num_gpus=0,
            max_concurrent_trials=1,
            num_models=num_models,
        )
        self.assertEqual(len(auto_trainer.training_results.errors), 0)
        self.assertEqual(len(auto_trainer.training_results), num_models)


if __name__ == "__main__":
    unittest.main()
