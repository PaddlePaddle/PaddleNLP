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

import paddle

from paddlenlp.transformers import BertForQuestionAnswering
from paddlenlp.utils.log import logger

from .benchmark_utils import rand_int_tensor
from .model_base import BenchmarkBase


class BertForQuestionAnsweringBenchmark(BenchmarkBase):
    def __init__(self):
        self.label_list = None
        self.pad_token_id = 0
        super().__init__()

    @staticmethod
    def add_args(args, parser):
        parser.add_argument(
            "--model_name_or_path", type=str, default="bert-base-cased", help="Model name. Defaults to bert-base. "
        )
        # args.max_seq_len

    def create_data_loader(self, args, **kwargs):
        raise NotImplementedError(
            "bert_for_question_answering's DataLoader is not implemented. Please use --generated_inputs. "
        )

    def generate_inputs_for_model(self, args, model, **kwargs):
        input_ids = rand_int_tensor(1, model.config.vocab_size, [args.batch_size, args.max_seq_len])
        start_positions = rand_int_tensor(
            0,
            args.max_seq_len,
            [
                args.batch_size,
            ],
        )
        end_positions = rand_int_tensor(
            0,
            args.max_seq_len,
            [
                args.batch_size,
            ],
        )
        return {"input_ids": input_ids, "start_positions": start_positions, "end_positions": end_positions}

    def build_model(self, args, **kwargs):
        model = BertForQuestionAnswering.from_pretrained(args.model_name_or_path)
        self.pad_token_id = model.config.pad_token_id
        return model

    def forward(self, model, args, input_data=None, **kwargs):
        start_positions = input_data.pop("start_positions")
        end_positions = input_data.pop("end_positions")
        start_logits, end_logits = model(**input_data)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if start_positions.ndim > 1:
                start_positions = start_positions.squeeze(-1)
            if start_positions.ndim > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clip(0, ignored_index)
            end_positions = end_positions.clip(0, ignored_index)

            loss_fct = paddle.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return (
            total_loss,
            paddle.sum((input_data["input_ids"] != self.pad_token_id)).numpy().astype("int64").item(),
        )

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
