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

import os
import sys

import paddle

from paddlenlp.transformers.xlnet.modeling import XLNetForSequenceClassification
from paddlenlp.transformers.xlnet.tokenizer import XLNetTokenizer
from paddlenlp.utils.log import logger

from .model_base import BenchmarkBase

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir, "examples", "language_model"
        )
    )
)
from xlnet.run_glue import create_data_loader  # noqa: E402


class XLNetBenchmark(BenchmarkBase):
    def __init__(self):
        self.label_list = None
        super().__init__()

    @staticmethod
    def add_args(args, parser):
        parser.add_argument(
            "--model_name_or_path",
            type=str,
            default="xlnet-base-cased",
            help="Model name. Defaults to xlnet-base-cased. ",
        )
        parser.add_argument("--task_name", type=str, default="SST-2", help="Task name. Defaults to sst-2. ")
        parser.add_argument("--max_seq_length", type=int, default=args.max_seq_len, help="Maximum sequence length. ")

    def create_data_loader(self, args, **kwargs):
        args.task_name = args.task_name.lower()
        tokenizer = XLNetTokenizer.from_pretrained(args.model_name_or_path)

        if args.task_name == "mnli":
            (
                train_data_loader,
                dev_data_loader_matched,
                dev_data_loader_mismatched,
                train_ds,
                _,
                _,
            ) = create_data_loader(args, tokenizer)
        else:
            train_loader, dev_loader, train_ds, _ = create_data_loader(args, tokenizer)

        self.num_batch = len(train_loader)
        self.label_list = train_ds.label_list

        if args.task_name == "mnli":
            return train_data_loader, (dev_data_loader_matched, dev_data_loader_mismatched)
        else:
            return train_loader, dev_loader

    def build_model(self, args, **kwargs):
        num_classes = 1 if self.label_list is None else len(self.label_list)
        model = XLNetForSequenceClassification.from_pretrained(args.model_name_or_path, num_classes=num_classes)

        self.loss_fct = paddle.nn.loss.CrossEntropyLoss() if self.label_list else paddle.nn.loss.MSELoss()

        return model

    def forward(self, model, args, input_data=None, **kwargs):
        input_ids, token_type_ids, attention_mask, labels = input_data
        logits = model(input_ids, token_type_ids, attention_mask)
        loss = self.loss_fct(logits, labels)

        return loss, args.batch_size

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
        max_mem_reserved_msg = ""
        max_mem_allocated_msg = ""
        if paddle.device.is_compiled_with_cuda():
            max_mem_reserved_msg = f"max_mem_reserved: {paddle.device.cuda.max_memory_reserved() // (1024 ** 2)} MB,"
            max_mem_allocated_msg = f"max_mem_allocated: {paddle.device.cuda.max_memory_allocated() // (1024 ** 2)} MB"
        logger.info(
            "global step %d / %d, loss: %f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, "
            "avg_samples: %.5f, ips: %.5f sequences/sec, %s %s"
            % (
                step_id,
                args.epoch * self.num_batch,
                loss,
                reader_cost,
                batch_cost,
                num_samples,
                ips,
                max_mem_reserved_msg,
                max_mem_allocated_msg,
            )
        )
