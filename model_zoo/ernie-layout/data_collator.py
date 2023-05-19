# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The HuggingFace Inc. team.
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

from typing import Optional, Union
from dataclasses import dataclass

from paddlenlp.transformers.tokenizer_utils_base import PretrainedTokenizerBase, PaddingStrategy


@dataclass
class DataCollator:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PretrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    label_pad_token_id: int = -100
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "np"

    def __call__(self, features):
        has_labels = "labels" in features[0]
        for feat in features:
            feat["input_ids"] = feat["input_ids"] + [1 * self.tokenizer.tokens_to_ids[self.tokenizer.pad_token]] * (
                self.max_length - len(feat["input_ids"])
            )
            feat["attention_mask"] = feat["attention_mask"] + [
                1 * self.tokenizer.tokens_to_ids[self.tokenizer.pad_token]
            ] * (self.max_length - len(feat["attention_mask"]))
            feat["bbox"] = feat["bbox"] + [[0, 0, 0, 0] for _ in range(self.max_length - len(feat["bbox"]))]
            if has_labels and not isinstance(feat["labels"], int):
                feat["labels"] = feat["labels"] + [1 * self.label_pad_token_id] * (
                    self.max_length - len(feat["labels"])
                )

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors=self.return_tensors,
        )
        return batch
