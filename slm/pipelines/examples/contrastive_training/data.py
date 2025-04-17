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

import math
import os.path
import random
from dataclasses import dataclass

import datasets
from arguments import DataArguments
from paddle.io import Dataset

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.transformers import PretrainedTokenizer


class TrainDatasetForEmbedding(Dataset):
    def __init__(
        self,
        args: DataArguments,
        tokenizer: PretrainedTokenizer,
        query_max_len: int = 64,
        passage_max_len: int = 1048,
        is_batch_negative: bool = False,
    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset(
                    "json",
                    data_files=os.path.join(args.train_data, file),
                    split="train",
                )
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(
                            list(range(len(temp_dataset))),
                            args.max_example_num_per_dataset,
                        )
                    )
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset("json", data_files=args.train_data, split="train")
        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len
        self.is_batch_negative = is_batch_negative

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        query = self.dataset[item]["query"]
        if self.args.query_instruction_for_retrieval is not None:
            query = self.args.query_instruction_for_retrieval + query
        query = self.tokenizer(
            query,
            truncation=True,
            max_length=self.query_max_len,
            return_attention_mask=False,
            truncation_side="right",
        )
        passages = []
        pos = random.choice(self.dataset[item]["pos"])
        passages.append(pos)
        # Add negative examples
        if not self.is_batch_negative:
            if len(self.dataset[item]["neg"]) < self.args.train_group_size - 1:
                num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]["neg"]))
                negs = random.sample(self.dataset[item]["neg"] * num, self.args.train_group_size - 1)
            else:
                negs = random.sample(self.dataset[item]["neg"], self.args.train_group_size - 1)
            passages.extend(negs)

        if self.args.passage_instruction_for_retrieval is not None:
            passages = [self.args.passage_instruction_for_retrieval + p for p in passages]
        passages = self.tokenizer(
            passages,
            truncation=True,
            max_length=self.passage_max_len,
            return_attention_mask=False,
            truncation_side="right",
        )
        # Convert passages to input_ids
        passages_tackle = []
        for i in range(len(passages["input_ids"])):
            passages_tackle.append({"input_ids": passages["input_ids"][i]})
        return query, passages_tackle


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    query_max_len: int = 32
    passage_max_len: int = 128

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]
        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])
        q_collated = self.tokenizer.pad(
            query,
            padding="max_length",
            max_length=self.query_max_len,
            return_attention_mask=True,
            pad_to_multiple_of=None,
            return_tensors="pd",
        )
        d_collated = self.tokenizer.pad(
            passage,
            padding="max_length",
            max_length=self.passage_max_len,
            return_attention_mask=True,
            pad_to_multiple_of=None,
            return_tensors="pd",
        )
        return {"query": q_collated, "passage": d_collated}
