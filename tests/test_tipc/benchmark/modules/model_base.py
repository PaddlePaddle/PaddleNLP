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

from paddlenlp.utils.log import logger


class BenchmarkBase(object):
    def __init__(self):
        self.num_batch = 0

    @staticmethod
    def add_args(args, parser):
        parser = parser.add_argument_group()

    def create_data_loader(self, args, **kwargs):
        raise NotImplementedError

    def build_model(self, args, **kwargs):
        raise NotImplementedError

    def generate_inputs_for_model(self, args, **kwargs):
        raise NotImplementedError

    def create_input_specs(self):
        return None

    def forward(self, model, args, input_data=None, **kwargs):
        raise NotImplementedError

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
            "global step %d / %d, loss: %f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, avg_samples: %.5f, ips: %.5f sequences/sec"
            % (step_id, args.epoch * self.num_batch, loss, reader_cost, batch_cost, num_samples, ips)
        )
