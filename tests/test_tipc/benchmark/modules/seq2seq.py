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

from paddlenlp.utils.log import logger

from .model_base import BenchmarkBase

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir, "examples", "machine_translation"
        )
    )
)
from seq2seq.data import create_train_loader  # noqa: E402
from seq2seq.seq2seq_attn import CrossEntropyCriterion, Seq2SeqAttnModel  # noqa: E402


class Seq2SeqBenchmark(BenchmarkBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def add_args(args, parser):
        parser.add_argument("--num_layers", type=int, default=2, help="Number of layers. ")
        parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size. ")
        parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate. ")
        parser.add_argument("--init_scale", type=float, default=0.1, help="Initial scale. ")
        parser.add_argument("--max_len", type=int, default=args.max_seq_len, help="Number of layers. ")

    def create_data_loader(self, args, **kwargs):
        (train_loader, eval_loader, self.src_vocab_size, self.tgt_vocab_size, self.eos_id) = create_train_loader(args)

        self.num_batch = len(train_loader)

        return train_loader, eval_loader

    def build_model(self, args, **kwargs):
        model = Seq2SeqAttnModel(
            self.src_vocab_size,
            self.tgt_vocab_size,
            args.hidden_size,
            args.hidden_size,
            args.num_layers,
            args.dropout,
            self.eos_id,
        )

        self.criterion = CrossEntropyCriterion()

        return model

    def forward(self, model, args, input_data=None, **kwargs):
        (src, src_length, trg, label, trg_mask) = input_data
        predict = model(src, src_length, trg)
        loss = self.criterion(predict, label, trg_mask)

        return loss, paddle.sum(trg_mask).numpy()

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
            "global step %d / %d, loss: %.6f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, "
            "avg_samples: %.5f, ips: %.5f words/sec, %s %s"
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
