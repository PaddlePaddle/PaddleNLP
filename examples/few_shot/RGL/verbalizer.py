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
from typing import List, Dict, Union, Callable, Optional
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.utils.log import logger
from paddlenlp.transformers import PretrainedTokenizer

from data import InputExample


class Verbalizer(nn.Layer):
    """
    Base verbalizer class used to process the outputs and labels.

    Args:
        tokenizer (paddlenlp.transformers.PretrainedTokenizer):
            The tokenizer of pretrained models.
        labels (list):
            The sequence of labels in task.
    
    """

    def __init__(self,
                 tokenizer: PretrainedTokenizer = None,
                 labels: List = None):
        super().__init__()
        assert labels is not None, 'Label list for current task is not set yet.'
        self.tokenizer = tokenizer
        self.labels = sorted(labels)
        self._process_lock = False

    @property
    def vocab(self):
        if not hasattr(self, '_vocab'):
            self._vocab = self.tokenizer.convert_ids_to_tokens(
                np.arange(self.vocab_size).tolist())
        return self._vocab

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def label_to_words(self):
        if not hasattr(self, '_label_to_words'):
            raise RuntimeError(
                'Property label_to_words has not been set before used.')
        return self._label_to_words

    @label_to_words.setter
    def label_to_words(self, label_to_words: Union[List, Dict]):
        if label_to_words is None:
            return
        if isinstance(label_to_words, dict):
            new_keys = sorted(list(label_to_words.keys()))
            assert new_keys == self.labels, 'label_to_words {} does not match the predefined labels {}.'.format(
                new_keys, self.labels)
            self._label_to_words = {k: label_to_words[k] for k in self.labels}
        elif isinstance(label_to_words, list):
            assert len(self.labels) == len(
                label_to_words
            ), 'The lengths of label_to_words and predefined labels do not match.'
            self._label_to_words = {
                k: v
                for k, v in zip(self.labels, label_to_words)
            }
        else:
            raise TypeError('Unsupported type {} for label_to_words'.format(
                type(label_to_words)))
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

    @abstractmethod
    def process_label_words(self, ):
        """ A hook to process verbalizer when it is set. """
        raise NotImplementedError

    @abstractmethod
    def project(self, logits, **kwargs):
        """ 
        Project the logits with shape ```[batch_size, vocab_size]``` into
        label_word_logits with shape ```[batch_size, num_label_words]```.
        """
        raise NotImplementedError

    @staticmethod
    def aggregate(label_words_logits, atype='mean', ndim=2):
        """
        Aggregate embeddings when multiple words are mapped to one label.
        
        Args:
            label_words_logits (paddle.Tensor):
                The logits of words which could be mapped to labels.
            atype (str):
                The aggregation strategy, including mean and first.
            ndim (str):
                The aggregated embeddings' number of dimensions.

        """
        if label_words_logits.ndim > ndim:
            if atype == 'mean':
                return label_words_logits.mean(axis=-1)
            elif atype == 'max':
                return label_words_logits.max(axis=-1)
            elif atype == 'first':
                return label_words_logits[..., 0, :]
            else:
                raise ValueError('Unsupported aggreate type {}'.format(atype))
        return label_words_logits

    def normalize(self, logits):
        """ Normalize the logits of every example. """
        new_logits = F.softmax(logits.reshape(logits.shape[0], -1), axis=-1)
        return new_logits.reshape(*logits.shape)


class ManualVerbalizer(Verbalizer):
    """
    Manual Verbalizer to map labels to words for hard prompt methods.

    Args:
        tokenizer (paddlenlp.transformers.PretrainedTokenizer):
            The tokenizer of pretrained models.
        labels (list):
            The sequence of all labels.
        label_to_words (dict or list):
            The dictionary or corresponding list to map labels to words.
        prefix (str):
            The prefix string of words, used in PLMs like RoBERTa, which is sensitive to the prefix.
    """

    def __init__(self, tokenizer, labels=None, label_to_words=None, prefix=''):
        super().__init__(tokenizer=tokenizer, labels=labels)
        self.tokenizer = tokenizer
        self.labels = labels
        self.prefix = prefix
        self.label_to_words = label_to_words

    def process_label_words(self):
        word_ids = []
        for label in self.labels:
            word_ids.append(
                self.tokenizer.encode(self.prefix + self._label_to_words[label],
                                      add_special_tokens=False,
                                      return_token_type_ids=False)['input_ids'])
        self.word_ids = paddle.to_tensor(word_ids, dtype='int64').squeeze()
        self.label_to_words_ids = {k: v for k, v in zip(self.labels, word_ids)}

    def process_logits(self, logits, mask_ids=None, **kwargs):
        if mask_ids is not None:
            logits = logits[mask_ids == 1]
        label_words_logits = logits.index_select(index=self.word_ids, axis=-1)
        return label_words_logits

    def wrap_one_example(self, example):
        """ Process labels in InputExample According to the predefined verbalizer. """
        if isinstance(example, InputExample):
            try:
                example.label = self.labels_to_ids[example.cls_label]
            except KeyError:
                # Regression tasks.
                example.label = eval(example.cls_label)
            return example
        else:
            raise TypeError('InputExample')
