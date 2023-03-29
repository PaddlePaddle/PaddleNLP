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

from functools import partial

import paddle
from paddle.io import DataLoader, DistributedBatchSampler

from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import ErnieForSequenceClassification, ErnieTokenizer
from paddlenlp.utils.log import logger

from .model_base import BenchmarkBase


# Data pre-process function for clue benchmark datatset
def seq_convert_example(example, label_list, tokenizer=None, max_seq_length=512, **kwargs):
    """convert a glue example into necessary features"""
    is_test = False
    if "label" not in example.keys():
        is_test = True

    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        example["label"] = int(example["label"]) if label_dtype != "float32" else float(example["label"])
        label = example["label"]
    # Convert raw text to feature
    if "keyword" in example:  # CSL
        sentence1 = " ".join(example["keyword"])
        example = {"sentence1": sentence1, "sentence2": example["abst"], "label": example["label"]}
    elif "target" in example:  # wsc
        text, query, pronoun, query_idx, pronoun_idx = (
            example["text"],
            example["target"]["span1_text"],
            example["target"]["span2_text"],
            example["target"]["span1_index"],
            example["target"]["span2_index"],
        )
        text_list = list(text)
        assert text[pronoun_idx : (pronoun_idx + len(pronoun))] == pronoun, "pronoun: {}".format(pronoun)
        assert text[query_idx : (query_idx + len(query))] == query, "query: {}".format(query)
        if pronoun_idx > query_idx:
            text_list.insert(query_idx, "_")
            text_list.insert(query_idx + len(query) + 1, "_")
            text_list.insert(pronoun_idx + 2, "[")
            text_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
        else:
            text_list.insert(pronoun_idx, "[")
            text_list.insert(pronoun_idx + len(pronoun) + 1, "]")
            text_list.insert(query_idx + 2, "_")
            text_list.insert(query_idx + len(query) + 2 + 1, "_")
        text = "".join(text_list)
        example["sentence"] = text

    if tokenizer is None:
        return example
    if "sentence" in example:
        example = tokenizer(
            example["sentence"], max_length=max_seq_length, truncation=True, padding="max_length", return_tensors="np"
        )
    elif "sentence1" in example:
        example = tokenizer(
            example["sentence1"],
            text_pair=example["sentence2"],
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

    if not is_test:
        if "token_type_ids" in example:
            return {
                "input_ids": example["input_ids"][0],
                "token_type_ids": example["token_type_ids"][0],
                "labels": label,
            }
        else:
            return {"input_ids": example["input_ids"][0], "labels": label}
    else:
        return {"input_ids": example["input_ids"][0], "token_type_ids": example["token_type_ids"][0]}


class Ernie3ForSequenceClassificationBenchmark(BenchmarkBase):
    def __init__(self):
        super().__init__()
        self.pad_token_id = 0

    @staticmethod
    def add_args(args, parser):
        parser.add_argument(
            "--model_name_or_path",
            type=str,
            default="ernie-3.0-base-zh",
            help="Model name. Defaults to ernie-3.0-base-zh. ",
        )
        parser.add_argument(
            "--task_name",
            default="tnews",
            type=str,
            help="The name of the task to train selected in the list: afqmc, tnews, iflytek, ocnli, cmnli, cluewsc2020, csl",
        )
        parser.add_argument("--max_seq_length", type=int, default=args.max_seq_len, help="Maximum sequence length. ")

    def create_input_specs(self):
        input_ids = paddle.static.InputSpec(name="input_ids", shape=[-1, -1], dtype="int64")
        token_type_ids = paddle.static.InputSpec(name="token_type_ids", shape=[-1, -1], dtype="int64")
        labels = paddle.static.InputSpec(name="labels", shape=[-1], dtype="int64")
        return [input_ids, token_type_ids, None, None, None, labels]

    def create_data_loader(self, args, **kwargs):
        args.task_name = args.task_name.lower()

        tokenizer = ErnieTokenizer.from_pretrained(args.model_name_or_path)
        train_ds, _ = load_dataset("clue", args.task_name, splits=("train", "dev"))

        trans_func = partial(
            seq_convert_example, label_list=train_ds.label_list, tokenizer=tokenizer, max_seq_len=args.max_seq_length
        )

        train_ds = train_ds.map(trans_func, lazy=False)
        repeat_data = []
        for i in range(10):
            repeat_data.extend(train_ds.new_data)
        train_ds.new_data = repeat_data
        train_batch_sampler = DistributedBatchSampler(
            train_ds, batch_size=args.batch_size, shuffle=False, drop_last=True
        )
        # fix develop bug, we donot use DataCollatorWithPadding.
        # batchify_fn = DataCollatorWithPadding(tokenizer)

        train_loader = DataLoader(
            dataset=train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=4,  # when paddlepaddle<=2.4.1, if we use dynamicTostatic mode, we need set num_workeks > 0
        )

        self.num_batch = len(train_loader)

        return train_loader, None

    def build_model(self, args, **kwargs):
        train_ds = load_dataset("clue", args.task_name, splits="train")
        num_labels = 1 if train_ds.label_list is None else len(train_ds.label_list)
        model = ErnieForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        self.pad_token_id = model.config.pad_token_id
        return model

    def forward(self, model, args, input_data=None, **kwargs):
        loss = model(**input_data)[0]
        return loss, args.batch_size * args.max_seq_length

    def logger(
        self,
        args,
        step_id=None,
        pass_id=None,
        batch_id=None,
        loss=None,
        batch_cost=None,
        reader_cost=None,
        num_samples=None,
        ips=None,
        **kwargs
    ):
        logger.info(
            "global step %d / %d, loss: %f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, avg_samples: %.5f, ips: %.5f words/sec"
            % (step_id, args.epoch * self.num_batch, loss, reader_cost, batch_cost, num_samples, ips)
        )
