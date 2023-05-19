"""
Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This module defines the itermediate data structure of inputs.
"""

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
from paddle import Tensor

from ..transformers.model_outputs import MaskedLMOutput, SequenceClassifierOutput
from ..transformers.tokenizer_utils_base import PaddingStrategy, PretrainedTokenizerBase


def signature(function):
    """
    Obtain the input arguments of the given function.
    """
    sig = inspect.signature(function)
    args = [p.name for p in sig.parameters.values() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
    return args


@dataclass
class PromptDataCollatorWithPadding:
    """
    Data collator that will group inputs by keywords and dynamically
    pad the inputs to the longest sequence in the batch.

    Args:
        tokenizer (`paddlenlp.transformers.PretrainedTokenizer`):
            The tokenizer used for encoding the data from PromptTokenizer.
    """

    tokenizer: PretrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pd"
    return_attention_mask: Optional[bool] = None
    default_model_input_names: List = (
        "input_ids",
        "token_type_ids",
        "special_tokens_mask",
        "offset_mapping",
        "position_ids",
    )

    def _convert_to_tensors(self, data):
        if self.return_tensors == "np":
            return np.array(data)
        else:
            return paddle.to_tensor(data)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        for key in features[0]:
            if key in self.default_model_input_names:
                batch[key] = [b[key] for b in features]

        batch = self.tokenizer.pad(
            batch,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
            return_attention_mask=self.return_attention_mask,
        )
        max_length = batch["input_ids"].shape[1]
        for key in features[0]:
            if key not in self.default_model_input_names:
                values = [b[key] for b in features if key in b]
                if len(values) < len(features):
                    continue
                if key == "masked_positions":
                    new_values = []
                    for index, value in enumerate(values):
                        value = np.array(value) + index * max_length
                        new_values.extend(value.tolist())
                    values = new_values
                elif key == "attention_mask":
                    new_values = np.ones([len(values), 1, max_length, max_length]) * -1e4
                    for index, value in enumerate(values):
                        length = len(value)
                        new_values[index][0, :length, :length] = value
                    values = new_values
                elif key in ("soft_token_ids", "encoder_ids"):
                    for index, value in enumerate(values):
                        values[index] = value + [0] * (max_length - len(value))
                elif key in ("omask_positions"):
                    max_num_option = max([len(x) for x in values])
                    for index, value in enumerate(values):
                        values[index] = value + [0] * (max_num_option - len(value))
                elif key == "labels":
                    if isinstance(values[0], list):
                        max_num_label = max([len(x) for x in values])
                        for index, value in enumerate(values):
                            values[index] = value + [-100] * (max_num_label - len(value))
                elif key != "cls_positions":
                    continue
                batch[key] = self._convert_to_tensors(values)
        return batch


def sequence_classification_forward_with_past_key_values(
    self,
    input_ids: Optional[Tensor] = None,
    token_type_ids: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    inputs_embeds: Optional[Tensor] = None,
    labels: Optional[Tensor] = None,
    output_hidden_states: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
):
    outputs = self.ernie(
        input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        past_key_values=past_key_values,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    pooled_output = outputs[1]

    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)

    loss = None
    if labels is not None:
        if self.num_labels == 1:
            loss_fct = paddle.nn.MSELoss()
            loss = loss_fct(logits, labels)
        elif labels.dtype == paddle.int64 or labels.dtype == paddle.int32:
            loss_fct = paddle.nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape((-1, self.num_labels)), labels.reshape((-1,)))
        else:
            loss_fct = paddle.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

    return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def masked_lm_forward_with_past_key_values(
    self,
    input_ids: Optional[Tensor] = None,
    token_type_ids: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    masked_positions: Optional[Tensor] = None,
    inputs_embeds: Optional[Tensor] = None,
    labels: Optional[Tensor] = None,
    output_hidden_states: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
):
    outputs = self.ernie(
        input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        past_key_values=past_key_values,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    sequence_output = outputs[0]
    prediction_scores = self.cls(sequence_output, masked_positions=masked_positions)

    masked_lm_loss = None
    if labels is not None:
        loss_fct = paddle.nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(
            prediction_scores.reshape((-1, paddle.shape(prediction_scores)[-1])), labels.reshape((-1,))
        )

    return MaskedLMOutput(
        loss=masked_lm_loss,
        logits=prediction_scores,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
