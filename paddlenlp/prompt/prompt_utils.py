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
from typing import Any, Dict, List, Union, Optional
from dataclasses import dataclass

import numpy as np
import paddle

from ..transformers.tokenizer_utils_base import PretrainedTokenizerBase, PaddingStrategy


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
        tokenizer (`paddlennlp.transformers.PretrainedTokenizer`):
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
                values = [b[key] for b in features]
                if key == "masked_positions":
                    new_values = []
                    for index, value in enumerate(values):
                        value = np.array(value) + index * max_length
                        new_values.extend(value.tolist())
                    values = new_values
                elif key == "attention_mask":
                    new_values = np.zeros([len(values), 1, max_length, max_length])
                    for index, value in enumerate(values):
                        length = len(value)
                        new_values[index][0, :length, :length] = value
                    values = new_values
                elif key in ("soft_token_ids", "encoder_ids"):
                    for index, value in enumerate(values):
                        values[index] = value + [0] * (max_length - len(value))
                elif key != "labels":
                    continue
                batch[key] = self._convert_to_tensors(values)
        return batch
