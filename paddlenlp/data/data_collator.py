# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.

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

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, NewType

import numpy as np
import paddle

from .collate import Stack, Pad

__all__ = [
    'DataCollatorWithPadding',
    'default_data_collator',
    'DataCollator',
    'DefaultDataCollator',
    'DataCollatorForTokenClassification',
    'DataCollatorForSeq2Seq',
]

InputDataClass = NewType("InputDataClass", Any)
"""
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of PyTorch/TensorFlow tensors or NumPy arrays.
"""
DataCollator = NewType("DataCollator",
                       Callable[[List[InputDataClass]], Dict[str, Any]])


def default_data_collator(data, return_tensors=True):

    if not isinstance(data[0], dict):
        data = [vars(f) for f in data]
    first = data[0]
    batch = {}

    if "label" in first and first["label"] is not None:
        label = first["label"]
        dtype = 'int64' if isinstance(label, int) else 'float32'
        batch["labels"] = Stack(dtype=dtype)([d["label"] for d in data])
    elif "label_ids" in first and first["label_ids"] is not None:
        dtype = 'int64' if type(first["label_ids"][0]) is int else 'float32'
        batch["labels"] = Stack(dtype=dtype)([d["label_ids"] for d in data])

    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(
                v, str):
            batch[k] = Stack(dtype='int64')([d[k] for d in data])
    if return_tensors:
        for k, v in batch.items():
            batch[k] = paddle.to_tensor(v)
    return batch


class DefaultDataCollator:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:
        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object
    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    This is an object (like other data collators) rather than a pure function like default_data_collator. This can be
    helpful if you need to set a return_tensors value at initialization.
    Args:
        return_tensors (`bool`):
            Return Tensor or numpy array.
    """
    return_tensors: bool = True

    def __call__(self, features: List[Dict[str, Any]],
                 return_tensors=None) -> Dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors
        return default_data_collator(features, return_tensors)


class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.

    Args:
        tokenizer (`paddlenlp.transformers.PretrainedTokenizer`):
            The tokenizer used for encoding the data.
    """

    def __init__(self, tokenizer, return_tensors=True):
        self.tokenizer = tokenizer
        self.return_tensors = return_tensors

    def __call__(self, data):
        first = data[0]
        assert isinstance(first, dict), 'Input pattern not understood. The input of collatot must be a dict with key of input column name and value of data ' \
                                   'Received input type:' % (type(first))
        batch = {}
        if "label" in first and first["label"] is not None:
            label = first["label"]
            dtype = 'int64' if isinstance(label, int) else 'float32'
            batch["labels"] = Stack(dtype=dtype)([d["label"] for d in data])
        elif "label_ids" in first and first["label_ids"] is not None:
            dtype = 'int64' if type(first["label_ids"][0]) is int else 'float32'
            batch["labels"] = Stack(dtype=dtype)([d["label_ids"] for d in data])

        for k, v in first.items():
            if k not in ("label", "label_ids"
                         ) and v is not None and not isinstance(v, str):
                if k == 'token_type_ids':
                    batch[k] = Pad(axis=0,
                                   pad_val=self.tokenizer.pad_token_type_id,
                                   dtype='int64')([d[k] for d in data])
                elif k == 'attention_mask':
                    batch[k] = Pad(axis=0, pad_val=0,
                                   dtype='int64')([d[k] for d in data])
                elif k == 'special_tokens_mask':
                    batch[k] = Pad(axis=0, pad_val=1,
                                   dtype='int64')([d[k] for d in data])
                elif k == 'input_ids':
                    batch[k] = Pad(axis=0,
                                   pad_val=self.tokenizer.pad_token_id,
                                   dtype='int64')([d[k] for d in data])
                else:
                    dtype = 'int64' if type(v) is int else 'float32'
                    batch[k] = Stack(dtype=dtype)([d[k] for d in data])

        if self.return_tensors:
            for k, v in batch.items():
                batch[k] = paddle.to_tensor(v)
        return batch


class DataCollatorForTokenClassification:
    """
    Data collator that will dynamically pad the inputs to longest sequence in the batch, as well as the labels.

    Args:
        tokenizer (`paddlenlp.transformers.PretrainedTokenizer`):
            The tokenizer used for encoding the data.
        label_pad_token_id (int, optional):
            The id to use when padding the labels. Defaults to -100.
    """

    def __init__(self, tokenizer, label_pad_token_id=-100, return_tensors=True):
        self.tokenizer = tokenizer
        self.return_tensors = return_tensors
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, data):
        first = data[0]
        assert isinstance(first, dict), 'Input pattern not understood. The input of collatot must be a dict with key of input column name and value of data ' \
                                   'Received input type:' % (type(first))
        batch = {}

        for k, v in first.items():
            if k not in ("label", "label_ids", "labels"
                         ) and v is not None and not isinstance(v, str):
                if k == 'token_type_ids':
                    batch[k] = Pad(axis=0,
                                   pad_val=self.tokenizer.pad_token_type_id,
                                   dtype='int64')([d[k] for d in data])
                elif k == 'attention_mask':
                    batch[k] = Pad(axis=0, pad_val=0,
                                   dtype='int64')([d[k] for d in data])
                elif k == 'special_tokens_mask':
                    batch[k] = Pad(axis=0, pad_val=1,
                                   dtype='int64')([d[k] for d in data])
                elif k == 'input_ids':
                    batch[k] = Pad(axis=0,
                                   pad_val=self.tokenizer.pad_token_id,
                                   dtype='int64')([d[k] for d in data])
                else:
                    batch[k] = Stack()([d[k] for d in data])
            else:
                batch[k] = Pad(axis=0,
                               pad_val=self.label_pad_token_id,
                               dtype='int64')([d[k] for d in data])
        if self.return_tensors:
            for k, v in batch.items():
                batch[k] = paddle.to_tensor(v)
        return batch


class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    def __init__(self,
                 tokenizer,
                 model=None,
                 label_pad_token_id=-100,
                 return_tensors=True):
        self.tokenizer = tokenizer
        self.model = model
        self.return_tensors = return_tensors
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, data, return_tensors=None):
        first = data[0]
        assert isinstance(first, dict), 'Input pattern not understood. The input of collatot must be a dict with key of input column name and value of data ' \
                                   'Received input type:' % (type(first))

        labels = [data["labels"]
                  for d in data] if "labels" in data[0].keys() else None

        batch = {}
        for k, v in first.items():
            if k not in ("labels", "label_ids"
                         ) and v is not None and not isinstance(v, str):
                if k == 'token_type_ids':
                    batch[k] = Pad(axis=0,
                                   pad_val=self.tokenizer.pad_token_type_id,
                                   dtype='int64')([d[k] for d in data])
                elif k == 'attention_mask':
                    batch[k] = Pad(axis=0, pad_val=0,
                                   dtype='int64')([d[k] for d in data])
                elif k == 'special_tokens_mask':
                    batch[k] = Pad(axis=0, pad_val=1,
                                   dtype='int64')([d[k] for d in data])
                elif k == 'input_ids':
                    batch[k] = Pad(axis=0,
                                   pad_val=self.tokenizer.pad_token_id,
                                   dtype='int64')([d[k] for d in data])
            else:
                batch[k] = Pad(axis=0,
                               pad_val=self.label_pad_token_id,
                               dtype='int64')([d[k] for d in data])
        # prepare decoder_input_ids
        if (labels is not None and self.model is not None and
                hasattr(self.model, "prepare_decoder_input_ids_from_labels")):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=paddle.to_tensor(batch["labels"]))
            if not return_tensors:
                batch["decoder_input_ids"] = decoder_input_ids.numpy()
        if self.return_tensors:
            for k, v in batch.items():
                batch[k] = paddle.to_tensor(v)

        return batch
