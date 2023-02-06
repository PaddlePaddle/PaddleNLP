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
import sys
from functools import partial

import paddle
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import ErnieForSequenceClassification, ErnieTokenizer
from paddlenlp.utils.log import logger

from .model_base import BenchmarkBase

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir, "model_zoo", "ernie-3.0")
    ),
)


from utils import seq_convert_example  # noqa: E402


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
        train_ds, dev_ds = load_dataset("clue", args.task_name, splits=("train", "dev"))
        trans_func = partial(
            seq_convert_example, label_list=train_ds.label_list, tokenizer=tokenizer, max_seq_len=args.max_seq_length
        )

        train_ds = train_ds.map(trans_func, lazy=True)
        train_batch_sampler = DistributedBatchSampler(
            train_ds, batch_size=args.batch_size, shuffle=False, drop_last=True
        )

        dev_ds = dev_ds.map(trans_func, lazy=True)
        dev_batch_sampler = BatchSampler(dev_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

        batchify_fn = DataCollatorWithPadding(tokenizer)

        train_loader = DataLoader(
            dataset=train_ds,
            batch_sampler=train_batch_sampler,
            collate_fn=batchify_fn,
            num_workers=4,  # when paddlepaddle<=2.4.1, if we use dynamicTostatic mode, we need set num_workeks > 0
            return_list=True,
        )
        dev_loader = DataLoader(
            dataset=dev_ds, batch_sampler=dev_batch_sampler, collate_fn=batchify_fn, num_workers=4, return_list=True
        )
        self.num_batch = len(train_loader)

        return train_loader, dev_loader

    def build_model(self, args, **kwargs):
        train_ds = load_dataset("clue", args.task_name, splits="train")
        num_labels = 1 if train_ds.label_list is None else len(train_ds.label_list)
        model = ErnieForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        self.pad_token_id = model.config.pad_token_id
        return model

    def forward(self, model, args, input_data=None, **kwargs):
        loss = model(**input_data)[0]
        return loss, paddle.sum((input_data["input_ids"] != self.pad_token_id)).numpy().astype("int64").item()

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
