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

from __future__ import absolute_import, division, print_function

import copy

import paddle
import paddle.distributed as dist
import ppfleetx.models.language_model.gpt as gpt
from paddle import LazyGuard
from paddle.static import InputSpec
from ppfleetx.core.module.basic_module import BasicModule
from ppfleetx.data.tokenizers import GPTTokenizer
from ppfleetx.utils.log import logger

from ...auto_utils import process_configs


class LanguageModuleAuto(BasicModule):
    def __init__(self, configs):
        self.nranks = dist.get_world_size()
        super(LanguageModuleAuto, self).__init__(configs)

        self.loss_fn = self.get_loss_fn()

    def process_configs(self, configs):
        configs = process_configs(configs)
        return configs

    def get_model_size(self, l, h, v, s):
        P = 12 * l * h * h * (1 + 13 / (12 * h) + (v + s) / (12 * l * h))
        logger.info("Model Size: {:.2f} B".format(P / 1000.0 / 1000.0 / 1000.0))


class GPTModuleAuto(LanguageModuleAuto):
    def __init__(self, configs):
        super(GPTModuleAuto, self).__init__(configs)

    def get_model(self):
        model_setting = copy.deepcopy(self.configs.Model)
        model_setting.pop("module")
        model_setting.pop("name")

        l = model_setting["num_layers"]
        h = model_setting["hidden_size"]
        v = model_setting["vocab_size"]
        s = self.configs.Data.Train.dataset.max_seq_len
        self.get_model_size(l, h, v, s)

        self.tokenizer = GPTTokenizer.from_pretrained("gpt2")

        with LazyGuard():
            model = gpt.GPTForPretrainingAuto(gpt.GPTModelAuto(**model_setting))
        return model

    def get_loss_fn(self):
        model_setting = copy.deepcopy(self.configs.Model)
        return gpt.GPTPretrainingCriterionAuto(model_setting["mesh"])


class GPTGenerationModuleAuto(BasicModule):
    def __init__(self, configs):
        self.configs = configs
        self.generation_cfgs = configs.Generation
        self.nranks = paddle.distributed.get_world_size()

        super().__init__(configs)

    def process_configs(self, configs):
        configs = process_configs(configs)
        return configs

    def get_model(self):
        model_setting = copy.deepcopy(self.configs.Model)
        model_setting.pop("module")
        model_setting.pop("name")

        with LazyGuard():
            model = gpt.GPTForGenerationAuto(gpt.GPTModelAuto(**model_setting), self.generation_cfgs)

        self.tokenizer = GPTTokenizer.from_pretrained("gpt2")

        self.generation_cfgs["max_dec_len"] = self.adjust_length_to_model(self.generation_cfgs["max_dec_len"], 512)

        self.generation_cfgs["bos_token_id"] = self.tokenizer.eos_token_id
        self.generation_cfgs["eos_token_id"] = self.tokenizer.eos_token_id
        self.generation_cfgs["pad_token_id"] = self.tokenizer.eos_token_id

        return model

    def adjust_length_to_model(self, length, max_sequence_length):
        if length < 0 or length > max_sequence_length:
            length = max_sequence_length
        return length

    def left_padding(self, inputs, pad_id, padding="longest"):
        assert "input_ids" in inputs, "input_ids should be in inputs!"
        max_length = 0
        for ids in inputs["input_ids"]:
            max_length = max(max_length, len(ids))

        def extend_max_lenth(value, max_length, to_pad_id):
            return [to_pad_id] * (max_length - len(value)) + value

        def extend_filed(name, max_length, to_pad_id):
            values = inputs[name]
            res = []
            for index, value in enumerate(values):
                res.append(extend_max_lenth(value, max_length, to_pad_id))
            inputs[name] = res

        extend_filed("input_ids", max_length, pad_id)
        if "attention_mask" in inputs:
            extend_filed("attention_mask", max_length, 0)
        if "position_ids" in inputs:
            extend_filed("position_ids", max_length, 0)

        return inputs

    def input_spec(self):
        return [InputSpec(shape=[None, None], name="input_ids", dtype="int64")]
