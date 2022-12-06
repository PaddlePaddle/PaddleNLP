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

import paddle.nn as nn

from .prompt_utils import signature


class PromptModelForSequenceClassification(nn.Layer):
    """
    PromptModel for classification tasks.
    """

    def __init__(self, model, template, verbalizer=None, freeze_plm: bool = False, freeze_dropout: bool = False):
        super(PromptModelForSequenceClassification, self).__init__()
        self.plm = model
        self.template = template
        self.verbalizer = verbalizer
        self.freeze_plm = freeze_plm
        self.freeze_dropout = freeze_dropout
        if self.freeze_plm:
            for param in self.plm.parameters():
                param.stop_gradient = True
            if self.freeze_dropout:
                self.plm.eval()
        self.forward_keys = signature(self.plm.forward)
        self._mask_token_id = self.template.tokenizer.mask_token_id
        self._pad_token_id = self.template.tokenizer.pad_token_id

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        masked_positions=None,
        soft_token_ids=None,
        encoder_ids=None,
        **kwargs
    ):
        input_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "masked_positions": masked_positions,
            "soft_token_ids": soft_token_ids,
            "attention_mask": attention_mask,
            "encoder_ids": encoder_ids,
        }
        input_dict = self.template.process_batch(input_dict)
        model_inputs = {k: input_dict[k] for k in input_dict if k in self.forward_keys}
        if "masked_positions" in model_inputs:
            model_inputs.pop("masked_positions")
        outputs = self.plm(**model_inputs)
        if self.verbalizer is not None:
            label_outputs = self.verbalizer.process_outputs(outputs, input_dict["masked_positions"])
        else:
            label_outputs = outputs

        if kwargs.pop("return_hidden_states", False):
            return label_outputs, outputs
        else:
            return label_outputs

    def prompt_parameters(self):
        """
        Get the parameters of template and verbalizer.
        """
        params = [p for p in self.template.parameters()]
        if self.verbalizer is not None:
            params += [p for p in self.verbalizer.parameters()]
        return params
