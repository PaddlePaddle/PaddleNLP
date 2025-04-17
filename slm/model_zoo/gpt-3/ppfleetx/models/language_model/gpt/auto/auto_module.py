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

import copy

import paddle
import ppfleetx.models.language_model.gpt as gpt
from paddle import LazyGuard
from paddle.static import InputSpec
from ppfleetx.core.module.basic_module import BasicModule
from ppfleetx.data.tokenizers import GPTTokenizer
from ppfleetx.utils.log import logger
from ppfleetx.models.language_model.language_module import vocab_size_with_padding

from paddlenlp.transformers.gpt.tokenizer import GPTChineseTokenizer

from ...auto_utils import process_configs
from ...language_module import get_model_size

MODEL_CLASSES = {
    "GPT": (GPTTokenizer, "gpt2"),
    "GPT-cn": (GPTChineseTokenizer, "gpt-cpm-large-cn"),
}


class LanguageModuleAuto(BasicModule):
    def __init__(self, configs):
        self.nranks = paddle.distributed.get_world_size()
        super(LanguageModuleAuto, self).__init__(configs)

        self.loss_fn = self.get_loss_fn()

    def process_configs(self, configs):
        configs = process_configs(configs)
        return configs

    def training_step_end(self, log_dict):
        speed = 1.0 / log_dict["train_cost"]
        default_global_tokens_num = self.configs.Global.global_batch_size * self.configs.Data.Train.dataset.max_seq_len

        loss_scale_str = ("loss_scale: %.9f," % (log_dict["loss_scale"]) if
                          log_dict.get("loss_scale", None) is not None else "")

        logger_info_str = "[train] epoch: [%d/%d], batch: [%d/%d]" \
                          % (
                            log_dict["epoch"],
                            log_dict["total_epoch"],
                            log_dict["batch"],
                            log_dict["total_step"],
                          )

        if "loss" in log_dict:
            logger_info_str += ", loss: %.9f" % (log_dict["loss"])

        logger_info_str += ", avg_batch_cost: %.5f sec, speed: %.2f step/s, ips_total: %.0f tokens/s, ips: %.0f tokens/s, %s learning rate: %.5e, found_inf: %.0f" \
                            % (
                                log_dict["train_cost"],
                                speed,
                                speed * default_global_tokens_num,
                                speed * default_global_tokens_num / log_dict["dp_world_size"],
                                loss_scale_str,
                                log_dict["lr"],
                                log_dict["found_inf"],
                            )

        if "max_memory_allocated" in log_dict:
            logger_info_str += ", max_memory_allocated: %.1f MB, max_memory_reserved: %.1f MB" \
                                ", memory_allocated: %.1f MB, memory_reserved: %.1f MB" \
                                % (log_dict["max_memory_allocated"], log_dict["max_memory_reserved"],log_dict["memory_allocated"], log_dict["memory_reserved"])

        logger.info(logger_info_str)

    def validation_step_end(self, log_dict):
        speed = 1.0 / log_dict["eval_cost"]

        logger_info_str = "[eval] epoch: %d, batch: %d/%d" \
                          % (
                            log_dict["epoch"],
                            log_dict["batch"],
                            log_dict["total_batch"],
                          )
        
        if "loss" in log_dict:
            logger_info_str += ", loss: %.9f" % (log_dict["loss"])

        logger_info_str += ", avg_eval_cost: %.5f sec, speed: %.2f step/s" \
                           % (
                            log_dict["eval_cost"],
                            speed,
                           )

        logger.info(logger_info_str)

    def test_step_end(self, log_dict):
        speed = 1.0 / log_dict["test_cost"]
        logger.info(
            "[test] epoch: %d, batch: %d, loss: %.9f, avg_test_cost: %.5f sec, speed: %.2f step/s"
            % (log_dict["epoch"], log_dict["batch"], log_dict["loss"],
               log_dict["test_cost"], speed))

    def training_epoch_end(self, log_dict):
        logger.info("[Training] epoch: %d, total time: %.5f sec" %
                    (log_dict["epoch"], log_dict["train_cost"]))


class GPTModuleAuto(LanguageModuleAuto):
    def __init__(self, configs):
        super(GPTModuleAuto, self).__init__(configs)

    def get_model(self):
        model_setting = copy.deepcopy(self.configs.Model)
        model_setting.pop("module")
        model_name = model_setting.pop("name")
        tokenizer_class, pretrained_name = MODEL_CLASSES[model_name]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_name)

        model_setting["vocab_size"] = vocab_size_with_padding(
            model_setting.get("vocab_size", self.tokenizer.vocab_size),
            model_setting.pop("vocab_size_divisible_unit", 128),
            self.configs.Distributed.get("mp_degree", 1),
        )

        l = model_setting["num_layers"]
        h = model_setting["hidden_size"]
        v = model_setting["vocab_size"]
        s = self.configs.Data.Train.dataset.max_seq_len
        get_model_size(l, h, v, s)

        with LazyGuard():
            model = gpt.GPTForPretrainingAuto(gpt.GPTModelAuto(**model_setting))
        return model

    def get_loss_fn(self):
        return gpt.GPTPretrainingCriterionAuto(sequence_parallel=self.configs.Model.sequence_parallel)


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
        model_name = model_setting.pop("name")
        tokenizer_class, pretrained_name = MODEL_CLASSES[model_name]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_name)

        model_setting["vocab_size"] = vocab_size_with_padding(
            model_setting.get("vocab_size", self.tokenizer.vocab_size),
            model_setting.pop("vocab_size_divisible_unit", 128),
            self.configs.Distributed.get("mp_degree", 1),
        )

        with LazyGuard():
            model = gpt.GPTForGenerationAuto(gpt.GPTModelAuto(**model_setting), self.generation_cfgs)

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
