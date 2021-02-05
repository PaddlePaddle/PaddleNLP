# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.transformers import *


class Ernie(nn.Layer):
    def __init__(self, model_name, num_classes, task=None, **kwargs):
        super().__init__()
        model_name = model_name.lower()
        self.task = task.lower()
        if self.task == 'seq-cls':
            required_names = list(ErnieForSequenceClassification.
                                  pretrained_init_configuration.keys())
            assert model_name in required_names, "model_name must be in %s, unknown %s ." (
                required_names, model_name)
            self.model = ErnieForSequenceClassification.from_pretrained(
                model_name, num_classes=num_classes, **kwargs)
        elif self.task == 'token-cls':
            required_names = list(ErnieForTokenClassification.
                                  pretrained_init_configuration.keys())
            assert model_name in required_names, "model_name must be in %s, unknown %s ." (
                required_names, model_name)
            self.model = ErnieForTokenClassification.from_pretrained(
                model_name, num_classes=num_classes, **kwargs)
        elif self.task == 'qa':
            required_names = list(
                ErnieForQuestionAnswering.pretrained_init_configuration.keys())
            assert model_name in required_names, "model_name must be in %s, unknown %s ." (
                required_names, model_name)
            self.model = ErnieForQuestionAnswering.from_pretrained(model_name,
                                                                   **kwargs)
        elif self.task is None:
            required_names = list(ErnieModel.pretrained_init_configuration.keys(
            ))
            assert model_name in required_names, "model_name must be in %s, unknown %s ." (
                required_names, model_name)
            self.model = ErnieModel.from_pretrained(model_name)
        else:
            raise RuntimeError(
                "Unknown task %s. Please make sure it to be one of seq-cls (it means sequence classifaction), "
                "token-cls (it means token classifaction), qa (it means question answering) "
                "or set it as None object." % task)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):

        if self.task in ['seq-cls', 'token-cls']:
            logits = self.model(input_ids, token_type_ids, position_ids,
                                attention_mask)
            return logits
        elif self.task == 'qa':
            start_logits, end_logits = self.model(input_ids, token_type_ids,
                                                  position_ids, attention_mask)
            start_position = paddle.unsqueeze(start_position, axis=-1)
            end_position = paddle.unsqueeze(end_position, axis=-1)
            return start_position, end_position
        elif self.task is None:
            sequence_output, pooled_output = self.model(
                input_ids, token_type_ids, position_ids, attention_mask)
            return sequence_output, pooled_output
