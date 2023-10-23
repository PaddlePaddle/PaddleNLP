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
from __future__ import annotations

import sys
import unittest
from functools import partial

from paddlenlp.transformers import AutoTokenizer


class ChatTemplateDataStreamTest(unittest.TestCase):
    data_file = "./tests/fixtures/llm/data/train.json"

    def setUp(self) -> None:
        self.root_path = "./llm"
        sys.path.insert(0, self.root_path)

    def tearDown(self) -> None:
        sys.path.remove(self.root_path)

    def test_load_dataset_for_train(self):
        from argument import DataArgument
        from data import convert_example_common, convert_examples_common
        from finetune_generation import load_dataset, read_local_dataset

        tokenizer = AutoTokenizer.from_pretrained("linly-ai/chinese-llama-2-13b")
        example_dataset = load_dataset(
            read_local_dataset,
            path=self.data_file,
            lazy=False,
        )
        example_data_args = DataArgument(use_chat_template=False)
        example_trans_func = partial(
            convert_example_common, tokenizer=tokenizer, data_args=example_data_args, is_test=False
        )
        example_dataset = example_dataset.map(example_trans_func)

        examples_dataset = load_dataset(
            read_local_dataset,
            path=self.data_file,
            lazy=False,
        )
        examples_data_args = DataArgument(use_chat_template=False)
        examples_trans_func = partial(
            convert_examples_common, tokenizer=tokenizer, data_args=examples_data_args, is_test=False
        )
        examples_dataset = examples_dataset.map(examples_trans_func)

        # 对比生成出来的 keys
        self.assertEqual(len(example_dataset), len(examples_dataset))

        for index in range(len(example_dataset)):
            example_item = example_dataset[index]
            examples_item = examples_dataset[index]
            print("example_item[input_ids]", example_item["input_ids"])
            print("examples_item[input_ids]", examples_item["input_ids"])
            # import pdb; pdb.set_trace()

            self.assertDictEqual(example_item, examples_item)
