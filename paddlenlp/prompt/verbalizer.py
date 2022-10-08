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

from abc import abstractmethod
from collections import defaultdict
import os
import copy
import json

import numpy as np
from typing import List, Dict, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..utils.log import logger

__all__ = [
    "Verbalizer", "MultiMaskVerbalizer", "ManualVerbalizer", "SoftVerbalizer"
]

VERBALIZER_FILE = "verbalizer.json"


class Verbalizer(nn.Layer):
    """
    Base verbalizer class used to process the outputs and labels.

    Args:
        labels (list):
            The sequence of labels in task.
    
    """

    def __init__(self, labels):
        super().__init__()
        self.labels = labels

    @property
    def labels(self):
        labels = getattr(self, "_labels", None)
        if labels is None:
            raise RuntimeError("`labels` is not set yet.")
        return labels

    @labels.setter
    def labels(self, labels):
        if labels is not None:
            self._labels = sorted(labels)
        else:
            self._labels = None

    @property
    def label_words(self):
        label_words = getattr(self, "_label_words", None)
        if label_words is None:
            raise RuntimeError("`label_words not set yet.")
        return label_words

    @label_words.setter
    def label_words(self, label_words: Union[List, Dict]):
        if label_words is None:
            return None
        if isinstance(label_words, dict):
            new_labels = sorted(list(label_words.keys()))
            if self._labels is None:
                self._labels = new_labels
            elif new_labels != self.labels:
                raise ValueError(
                    f"The given `label_words` {new_labels} does not match " +
                    f"predefined labels {self.labels}.")
            self._label_words = [label_words[k] for k in self.labels]
        elif isinstance(label_words, list):
            if self._labels is None:
                raise ValueError(
                    "`labels` should be set as the given `label_words` is "
                    "a list. Make sure that the order is compatible.")
            if len(self.labels) != len(label_words):
                raise ValueError(
                    "The length of given `label_words` and predefined " +
                    "labels do not match.")
            self._label_words = label_words
        else:
            raise TypeError('Unsupported type {} for label_words'.format(
                type(label_words)))
        self.process_label_words()

    @property
    def labels_to_ids(self):
        if not hasattr(self, 'labels'):
            raise RuntimeError(
                'Property labels_to_ids has not been set before used.')
        return {k: i for i, k in enumerate(self.labels)}

    @property
    def ids_to_labels(self):
        if not hasattr(self, 'labels'):
            raise RuntimeError(
                'Property ids_to_labels has not been set before used.')
        return {i: k for i, k in enumerate(self.labels)}

    @staticmethod
    def add_prefix(label_words, prefix):
        """ Add prefix to get expected token ids. """
        if isinstance(label_words[0], str):
            label_words = [[word] for word in label_words]

        new_label_words = []
        for words_per_label in label_words:
            new_words_per_label = []
            for word in words_per_label:
                new_words_per_label.append(prefix + word)
            new_label_words.append(new_words_per_label)
        return new_label_words

    @abstractmethod
    def process_label_words(self, ):
        """ A hook to process verbalizer when it is set. """
        raise NotImplementedError

    @abstractmethod
    def project(self, logits, **kwargs):
        """ 
        Project the logits with shape ```[..., vocab_size]``` into
        label_word_logits with shape ```[..., label_words]```.
        """
        raise NotImplementedError

    @staticmethod
    def aggregate(embeddings, mask=None, atype='mean', ndim=2):
        """
        Aggregate embeddings at the last dimension according to `atype`
        if its number of dimensions is greater than `ndim`.
        Used to handle multiple tokens for words and multiple words
        for labels.

        Args:
            embeddings (paddle.Tensor):
                The original embeddings.
            atype (str):
                The aggregation strategy, including mean and first.
            ndim (str):
                The aggregated embeddings' number of dimensions.

        """
        if embeddings.ndim > ndim and atype is not None:
            if atype == 'mean':
                if mask is None:
                    return embeddings.mean(axis=-1)
                return (embeddings * mask.unsqueeze(0)).sum(
                    axis=-1) / (mask.unsqueeze(0).sum(axis=-1) + 1e-10)
            elif atype == 'max':
                if mask is None:
                    return embeddings.max(axis=-1)
                return (embeddings - 1e4 * (1 - mask.unsqueeze(0))).max(axis=-1)
            elif atype == 'first':
                return embeddings[..., 0]
            else:
                raise ValueError('Unsupported aggregate type {}'.format(atype))
        return embeddings

    def normalize(self, logits):
        """ Normalize the logits of every example. """
        new_logits = F.softmax(logits.reshape(logits.shape[0], -1), axis=-1)
        return new_logits.reshape(*logits.shape)

    def from_file(self, path):
        """
        Load labels and corresponding words from files.
        """
        raise NotImplementedError

    def save_to(self, path):
        label_state = [self.labels, self.token_ids.numpy().tolist()]
        with open(os.path.join(path, VERBALIZER_FILE), "w") as f:
            json.dump(label_state, f)

    @classmethod
    def load_from(cls, path):
        with open(os.path.join(path, VERBALIZER_FILE), "r") as f:
            label_state = json.load(f)
            return label_state


class ManualVerbalizer(Verbalizer):
    """
    Manual Verbalizer to map labels to words for hard prompt methods.

    Args:
        tokenizer (paddlenlp.transformers.PretrainedTokenizer):
            The tokenizer of pretrained models.
        labels (list):
            The sequence of all labels.
        label_words (dict or list):
            The dictionary or corresponding list to map labels to words.
        prefix (str):
            The prefix string of words, used in PLMs like RoBERTa, which is sensitive to the prefix.
    """

    def __init__(self, tokenizer, labels=None, label_words=None, prefix=None):
        super().__init__(labels=labels)
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.label_words = label_words

    def process_label_words(self):
        """ Create the label-word-token array and its corresponding mask. """
        if self.prefix is not None:
            self._label_words = self.add_prefix(self.label_words, self.prefix)

        all_ids = []
        for words_per_label in self.label_words:
            word_ids = []
            for word in words_per_label:
                word_ids.append(
                    self.tokenizer.encode(
                        word,
                        add_special_tokens=False,
                        return_token_type_ids=False)["input_ids"])
            all_ids.append(word_ids)

        max_num_words = max([len(words) for words in self.label_words])
        max_num_tokens = max([
            max([len(token_ids) for token_ids in word_ids])
            for word_ids in all_ids
        ])
        token_ids_shape = [len(self.labels), max_num_words, max_num_tokens]
        token_ids = np.zeros(shape=token_ids_shape)
        token_mask = np.zeros(shape=token_ids_shape)
        word_mask = np.zeros(shape=[len(self.labels), max_num_words])
        for label_i, ids_per_label in enumerate(all_ids):
            word_mask[label_i][:len(ids_per_label)] = 1
            for word_i, ids_per_word in enumerate(ids_per_label):
                token_ids[label_i][word_i][:len(ids_per_word)] = ids_per_word
                token_mask[label_i][word_i][:len(ids_per_word)] = 1
        self.token_ids = paddle.to_tensor(token_ids,
                                          dtype="int64",
                                          stop_gradient=True)
        self.token_ids_mask = paddle.to_tensor(token_mask,
                                               dtype="int64",
                                               stop_gradient=True)
        self.word_ids_mask = paddle.to_tensor(word_mask,
                                              dtype="float32",
                                              stop_gradient=True)

    def project(self, logits):
        word_shape = [*logits.shape[:-1], *self.token_ids.shape]
        token_logits = logits.index_select(index=self.token_ids.reshape([-1]),
                                           axis=-1).reshape(word_shape)
        word_logits = self.aggregate(token_logits, self.token_ids_mask)
        return word_logits

    def process_outputs(self, logits, inputs, **kwargs):
        mask_ids = inputs["mask_ids"].unsqueeze(2)
        real_len = logits.shape[1]
        mask_ids = mask_ids[:, -real_len:]
        logits = paddle.where(mask_ids == 1, logits, paddle.zeros_like(logits))
        logits = logits.sum(axis=1) / mask_ids.sum(axis=1)

        word_logits = self.project(logits)
        label_logits = self.aggregate(word_logits, self.word_ids_mask)
        return label_logits

    @classmethod
    def from_file(cls, tokenizer, label_file, prefix=None, delimiter="=="):
        with open(label_file, "r", encoding="utf-8") as fp:
            label_words = defaultdict(list)
            for line in fp:
                data = line.strip().split(delimiter)
                word = data[1] if len(data) > 1 else data[0].split("##")[-1]
                label_words[data[0]].append(word)
        return cls(tokenizer,
                   labels=set(label_words.keys()),
                   label_words=dict(label_words),
                   prefix=prefix)


class MultiMaskVerbalizer(ManualVerbalizer):

    def __init__(self, tokenizer, labels=None, label_words=None, prefix=None):
        super().__init__(tokenizer, labels, label_words, prefix)

    def process_outputs(self, logits, inputs, **kwargs):
        """
        Process logits according to mask ids and label words.
        Args:
            logits (paddle.Tensor):
                 The output of ForMaskedLM model with shape
                 [batch_size, max_seq_length, vocab_size].
            inputs (InputFeatures):
                 The input features of model, including mask_ids.
        """
        batch_size, seq_len, vocab_size = logits.shape
        batch_ids, word_ids = paddle.where(inputs["mask_ids"] == 1)
        mask_ids = batch_ids * seq_len + word_ids
        mask_logits = logits.reshape([-1, vocab_size])[mask_ids]
        mask_logits = mask_logits.reshape([batch_size, -1, vocab_size])
        return mask_logits


class Identity(nn.Layer):
    """
    Identity layer to replace the last linear layer in MLM model, which
    outputs the input `sequence_output` directly.
    """

    def __init__(self):
        super().__init__()

    def forward(self, sequence_output, masked_positions=None):
        return sequence_output


class SoftVerbalizer(Verbalizer):
    """
    Soft Verbalizer to encode labels as embeddings.

    Args:
        tokenizer (paddlenlp.transformers.PretrainedTokenizer):
            The tokenizer of pretrained models.
        model (paddlenlp.transformers.PretrainedModel):
            The pretrained language model.
        labels (list):
            The sequence of all labels.
        label_words (dict or list):
            The dictionary or corresponding list to map labels to words.
        prefix (str):
            The prefix string of words, used in PLMs like RoBERTa, which is sensitive to the prefix.
    """

    LAST_WEIGHT = ["ErnieForMaskedLM", "BertForMaskedLM"]
    LAST_LINEAR = ["AlbertForMaskedLM", "RobertaForMaskedLM"]

    def __init__(self, tokenizer, model, labels, label_words=None, prefix=''):
        super().__init__(labels=labels)
        self.tokenizer = tokenizer
        self.labels = labels
        self.prefix = prefix
        self.label_words = label_words

        self._extract_head(model)

    def process_label_words(self):
        """ Create the label-token array and its corresponding mask. """
        if self.prefix is not None:
            self._label_words = self.add_prefix(self.label_words, self.prefix)

        all_ids = []
        for words_per_label in self.label_words:
            if len(words_per_label) > 1:
                logger.warning("Only the first word used for every label.")
            all_ids.append(
                self.tokenizer.encode(words_per_label[0],
                                      add_special_tokens=False,
                                      return_token_type_ids=False)["input_ids"])

        max_num_tokens = max([len(tokens) for tokens in all_ids])
        token_ids = np.zeros(shape=[len(self.labels), max_num_tokens])
        token_mask = np.zeros(shape=[len(self.labels), max_num_tokens])
        for label_i, ids_per_label in enumerate(all_ids):
            token_ids[label_i][:len(ids_per_label)] = ids_per_label
            token_mask[label_i][:len(ids_per_label)] = 1
        self.token_ids = paddle.to_tensor(token_ids,
                                          dtype="int64",
                                          stop_gradient=True)
        self.token_ids_mask = paddle.to_tensor(token_mask,
                                               dtype="int64",
                                               stop_gradient=True)

    def head_parameters(self):
        if isinstance(self.head, nn.Linear):
            return [(n, p) for n, p in self.head.named_parameters()]
        else:
            return [(n, p) for n, p in self.head.named_parameters()
                    if self.head_name[1] in n]

    def non_head_parameters(self):
        if isinstance(self.head, nn.Linear):
            return []
        else:
            return [(n, p) for n, p in self.head.named_parameters()
                    if self.head_name[1] not in n]

    def process_model(self, model):
        setattr(model, self.head_name[0], Identity())
        return model

    def process_outputs(self, logits, inputs=None, **kwargs):
        mask_ids = inputs["mask_ids"].unsqueeze(2)
        real_len = logits.shape[1]
        mask_ids = mask_ids[:, -real_len:]
        logits = (logits * mask_ids).sum(axis=1) / mask_ids.sum(axis=1)
        return self.head(logits)

    @classmethod
    def from_file(cls, tokenizer, model, label_file, prefix=None):
        with open(label_file, "r", encoding="utf-8") as fp:
            label_words = defaultdict(list)
            for line in fp:
                data = line.strip().split("==")
                word = data[1] if len(data) > 1 else data[0]
                label_words[data[0]].append(word)
        return cls(tokenizer,
                   model,
                   labels=set(label_words.keys()),
                   label_words=dict(label_words),
                   prefix=prefix)

    def _extract_head(self, model):
        model_type = model.__class__.__name__
        if model_type in self.LAST_LINEAR:
            # LMHead
            last_name = [n for n, p in model.named_children()][-1]
            self.head = copy.deepcopy(getattr(model, last_name))
            self.head_name = [last_name]
            head_names = [n for n, p in self.head.named_children()][::-1]
            for name in head_names:
                module = getattr(self.head, name)
                if isinstance(module, nn.Linear):
                    setattr(
                        self.head, name,
                        nn.Linear(module.weight.shape[0],
                                  len(self.labels),
                                  bias_attr=False))
                    getattr(self.head, name).weight.set_value(
                        self._create_init_weight(module.weight))
                    self.head_name.append(name)
                    break
        elif model_type in self.LAST_WEIGHT:
            # OnlyMLMHead
            last_name = [n for n, p in model.named_children()][-1]
            head = getattr(model, last_name)
            self.head_name = [last_name]
            # LMPredictionHead
            last_name = [n for n, p in head.named_children()][-1]
            self.head = copy.deepcopy(getattr(head, last_name))
            self.head_name.append("decoder")

            module = paddle.to_tensor(getattr(self.head, "decoder_weight"))
            bias = paddle.to_tensor(getattr(self.head, "decoder_bias"))
            new_head = nn.Linear(len(self.labels),
                                 module.shape[1],
                                 bias_attr=False)
            new_head.weight.set_value(self._create_init_weight(module.T).T)
            setattr(self.head, "decoder_weight", new_head.weight)
            getattr(self.head, "decoder_weight").stop_gradient = False
            setattr(
                self.head, "decoder_bias",
                self.head.create_parameter(shape=[len(self.labels)],
                                           dtype=new_head.weight.dtype,
                                           is_bias=True))
            getattr(self.head, "decoder_bias").stop_gradient = False
        else:
            raise NotImplementedError(
                f"Please open an issue to request for support of {model_type}" +
                f" or contribute to PaddleNLP.")

    def _create_init_weight(self, weight, is_bias=False):
        if is_bias:
            bias = paddle.index_select(weight,
                                       self.token_ids.reshape([-1]),
                                       axis=0).reshape(self.token_ids.shape)
            bias = self.aggregate(bias, self.token_ids_mask)
            return bias
        else:
            word_shape = [weight.shape[0], *self.token_ids.shape]
            weight = paddle.index_select(weight,
                                         self.token_ids.reshape([-1]),
                                         axis=1).reshape(word_shape)
            weight = self.aggregate(weight, self.token_ids_mask)
            return weight
