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
import paddle.distributed as dist

from paddlenlp.transformers import GPTForSequenceClassification
from paddlenlp.utils.log import logger

from .benchmark_utils import rand_int_tensor
from .model_base import BenchmarkBase


class GPTForSequenceClassificationBenchmark(BenchmarkBase):
    def __init__(self):
        self.label_list = None
        super().__init__()

    @staticmethod
    def add_args(args, parser):
        parser.add_argument(
            "--model_name_or_path", type=str, default="gpt2-en", help="Model name. Defaults to gpt2-en. "
        )
        # args.max_seq_len

    def create_data_loader(self, args, **kwargs):
        raise NotImplementedError(
            "gpt_for_sequence_classification's DataLoader is not implemented. Please use --generated_inputs. "
        )

    def create_input_specs(self):
        input_ids = paddle.static.InputSpec(name="input_ids", shape=[-1, -1], dtype="int64")
        labels = paddle.static.InputSpec(name="labels", shape=[-1], dtype="int64")
        return [input_ids, None, None, None, labels]

    def generate_inputs_for_model(self, args, model, **kwargs):
        input_ids = rand_int_tensor(1, model.config.vocab_size, [args.batch_size, args.max_seq_len])
        labels = rand_int_tensor(0, model.config.num_classes - 1, [args.batch_size])

        return {"input_ids": input_ids, "labels": labels}

    def build_model(self, args, **kwargs):
        model = GPTForSequenceClassification.from_pretrained(args.model_name_or_path)
        return model

    def forward(self, model, args, input_data=None, **kwargs):
        res = model(**input_data)
        if dist.get_world_size() == 1:
            pad_token_id = model.config.pad_token_id
        else:
            pad_token_id = model._layers.config.pad_token_id
        return (
            res[0],
            paddle.sum((input_data["input_ids"] != pad_token_id)).numpy().astype("int64").item(),
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
