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


from typing import Any, Dict, Optional

import paddle

from ..transformers.model_outputs import MaskedLMOutput, SequenceClassifierOutput
from .prompt_utils import signature
from .template import Template
from .verbalizer import Verbalizer


class PromptModelForSequenceClassification(paddle.nn.Layer):
    """
    PromptModel for classification tasks.
    """

    def __init__(
        self,
        model: paddle.nn.Layer,
        template: Template,
        verbalizer: Optional[Verbalizer] = None,
        freeze_plm: bool = False,
        freeze_dropout: bool = False,
    ):
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
        input_ids: paddle.Tensor,
        token_type_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        masked_positions: Optional[paddle.Tensor] = None,
        soft_token_ids: Optional[paddle.Tensor] = None,
        encoder_ids: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Dict[str, Any]
    ):
        return_dict = return_dict if return_dict is not None else False
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
        model_outputs = self.plm(**model_inputs, return_dict=True)
        if isinstance(model_outputs, MaskedLMOutput):
            if self.verbalizer is not None:
                logits = self.verbalizer.process_outputs(model_outputs.logits, input_dict["masked_positions"])
                num_labels = len(self.verbalizer.label_words)
            else:
                raise Exception("Verbalizer is required when model uses the MaskedLM head")
        elif isinstance(model_outputs, SequenceClassifierOutput):
            logits = model_outputs.logits
            num_labels = self.plm.num_classes if self.plm.num_classes is not None else self.plm.num_labels
        else:
            raise Exception(f"Model type not support yet: {type(model_outputs)}")

        loss = None
        if labels is not None:
            if num_labels == 1:
                loss_fct = paddle.nn.MSELoss()
                loss = loss_fct(logits, labels)
            elif labels.dtype == paddle.int64 or labels.dtype == paddle.int32:
                loss_fct = paddle.nn.CrossEntropyLoss()
                loss = loss_fct(logits.reshape((-1, num_labels)), labels.reshape((-1,)))
            else:
                loss_fct = paddle.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits, model_outputs.logits)
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=model_outputs.logits,
        )

    def prompt_parameters(self):
        """
        Get the parameters of template and verbalizer.
        """
        params = [p for p in self.template.parameters()]
        if self.verbalizer is not None:
            params += [p for p in self.verbalizer.parameters()]
        return params
