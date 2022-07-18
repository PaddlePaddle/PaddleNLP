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
import os
import copy
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..utils.log import logger
from ..transformers import PretrainedTokenizer

from .prompt_utils import InputExample

__all__ = ["Verbalizer", "ManualVerbalizer", "SoftVerbalizer"]


class Verbalizer(nn.Layer):
    """
    Base verbalizer class used to process the outputs and labels.

    Args:
        tokenizer (paddlenlp.transformers.PretrainedTokenizer):
            The tokenizer of pretrained models.
        labels (list):
            The sequence of labels in task.
    
    """

    def __init__(self, tokenizer, labels):
        super().__init__()
        self.tokenizer = tokenizer
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
    def label_to_words(self):
        label_to_words = getattr(self, "_label_to_words", None)
        if label_to_words is None:
            raise RuntimeError("`label_to_words not set yet.")
        return label_to_words

    @label_to_words.setter
    def label_to_words(self, label_to_words: Union[List, Dict]):
        if label_to_words is None:
            return None
        if isinstance(label_to_words, dict):
            new_labels = sorted(list(label_to_words.keys()))
            if new_labels != self.labels:
                raise ValueError(
                    f"The given `label_to_words` {new_labels} does not match " +
                    f"predefined labels {self.labels}.")
            self._label_to_words = {k: label_to_words[k] for k in self.labels}
        elif isinstance(label_to_words, list):
            if len(self.labels) != len(label_to_words):
                raise ValueError(
                    "The length of given `label_to_words` and predefined " +
                    "labels do not match.")
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

    def from_file(self, path, map_type="one-to-one"):
        """
        Load labels and corresponding words from files, which are formatted as
        label_name%%
        """
        if not os.path.isfile(path):
            raise ValueError(f"{path} is not a valid label file.")
        with open(path, 'w') as fp:
            lines = fp.readlines()
            label_to_words = {}
            if map_type == 'one-to-one':
                for line in lines:
                    label, word = lines.strip().split()
                    label_to_words[label] = word
            elif map_type == 'one-to-many':
                for line in lines:
                    label, word = lines.strip().split()
                    if label not in label_to_words:
                        label_to_words[label] = []
                    label_to_words[label].append(word)
            else:
                raise ValueError(f"Unsupported mapping type {map_type}.")

            self.labels = sorted(list(label_to_words.keys()))
            self.label_to_words = label_to_words


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

    def process_outputs(self, logits, inputs=None, **kwargs):
        if inputs is not None:
            mask_ids = inputs["mask_ids"]
            logits = logits[mask_ids == 1]
        label_words_logits = logits.index_select(index=self.word_ids, axis=-1)
        return label_words_logits

    def wrap_one_example(self, example):
        """ Process labels in InputExample According to the predefined verbalizer. """
        if isinstance(example, InputExample):
            example.labels = self.labels_to_ids[example.cls_label]
            return example
        else:
            raise TypeError('InputExample')


class SoftVerbalizer(ManualVerbalizer):
    """
    Soft Verbalizer to encode labels as embeddings.

    Args:
        tokenizer (paddlenlp.transformers.PretrainedTokenizer):
            The tokenizer of pretrained models.
        model (paddlenlp.transformers.PretrainedModel):
            The pretrained language model.
        labels (list):
            The sequence of all labels.
        label_to_words (dict or list):
            The dictionary or corresponding list to map labels to words.
        prefix (str):
            The prefix string of words, used in PLMs like RoBERTa, which is sensitive to the prefix.
    """

    def __init__(self,
                 tokenizer,
                 model,
                 labels,
                 label_to_words=None,
                 prefix=''):
        super().__init__(tokenizer=tokenizer, labels=labels)
        self.labels = labels
        self.prefix = prefix
        self.label_to_words = label_to_words

        head_name = [n for n, p in model.named_children()][-1]
        logger.info(f"The head module {head_name} will be retrieved.")
        self.head = copy.deepcopy(getattr(model, head_name))
        self.head_name = [head_name]
        if isinstance(self.head, nn.Linear):
            init_weight = paddle.index_select(self.head.weight,
                                              self.word_ids,
                                              axis=1)
            self.head = nn.Linear(self.head.weight.shape[0],
                                  len(self.labels),
                                  bias_attr=False)
            self.head.weight.set_value(init_weight)
        else:
            find_linear = False
            child_names = [n for n, p in self.head.named_children()][::-1]
            for name in child_names:
                module = getattr(self.head, name)
                if isinstance(module, nn.Linear):
                    self.head_name.append(name)
                    setattr(
                        self.head, name,
                        nn.Linear(module.weight.shape[0],
                                  len(self.labels),
                                  bias_attr=False))
                    getattr(self.head, name).weight.set_value(
                        paddle.index_select(module.weight,
                                            self.word_ids,
                                            axis=1))
                    find_linear = True
                    break
            if not find_linear:
                nested_head_name = [
                    n for n, p in self.head.named_children()
                    if 'Head' in p.__class__.__name__
                ][0]
                nested_head = getattr(self.head, nested_head_name)
                self.head_name.append(nested_head_name)
                nested_child_name = [n for n, p in nested_head.named_children()]
                for name in nested_child_name[::-1]:
                    module = getattr(nested_head, name)
                    if isinstance(module, nn.Linear):
                        self.head_name.append(name)
                        init_weight = paddle.index_select(module.weight,
                                                          self.word_ids,
                                                          axis=1)
                        setattr(
                            nested_head, name,
                            nn.Linear(module.weight.shape[0],
                                      len(self.labels),
                                      bias_attr=False))
                        getattr(nested_head, name).weight.set_value(init_weight)
                        find_linear = True
                        break
            if not find_linear:
                raise RuntimeError("Can not retrive Linear layer from PLM.")

    def head_parameters(self):
        if isinstance(self.head, nn.Linear):
            return [p for n, p in self.head.named_parameters()]
        else:
            head_name = '.'.join(self.head_name[1:])
            return [
                p for n, p in self.head.named_parameters() if head_name in n
            ]

    def non_head_parameters(self):
        if isinstance(self.head, nn.Linear):
            return []
        else:
            head_name = '.'.join(self.head_name[1:])
            return [
                p for n, p in self.head.named_parameters() if head_name not in n
            ]

    def process_model(self, model):
        setattr(model, self.head_name[0], nn.Identity())
        return model

    def process_outputs(self, logits, inputs=None, **kwargs):
        if inputs is not None:
            mask_ids = inputs["mask_ids"]
            logits = logits[mask_ids == 1]
        label_words_logits = logits.index_select(index=self.word_ids, axis=-1)
        return self.head(logits)
