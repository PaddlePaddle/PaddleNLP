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

import copy
import json
import os
from abc import abstractmethod
from typing import Dict

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor

from paddlenlp.transformers import PretrainedModel, PretrainedTokenizer
from paddlenlp.utils.log import logger

__all__ = ["Verbalizer", "ManualVerbalizer", "SoftVerbalizer", "MaskedLMVerbalizer"]

# Verbalizer used to be saved in a file.
VERBALIZER_CONFIG_FILE = "verbalizer_config.json"
VERBALIZER_PARAMETER_FILE = "verbalizer_state.pdparams"


class Verbalizer(nn.Layer):
    """
    Base class for [`Verbalizer`].

    Args:
        label_words (`dict`):
            Define the mapping from labels to a single or multiple words.
        tokenizer (`PretrainedTokenizer`):
            An instance of PretrainedTokenizer for label word tokenization.
    """

    def __init__(self, label_words: Dict, tokenizer: PretrainedTokenizer, **kwargs):
        super(Verbalizer, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.tokenizer = tokenizer
        self.token_aggregate_type = kwargs.get("token_aggregate_type", "mean")
        self.word_aggregate_type = kwargs.get("word_aggregate_type", "mean")
        self.mask_aggregate_type = kwargs.get("mask_aggregate_type", "product")
        self.post_log_softmax = kwargs.get("post_log_sigmoid", True)
        self.label_token_weight = kwargs.get("label_token_weight", None)
        if self.label_token_weight is not None:
            self.label_token_weight = self.normalize(self.project(self.label_token_weight.unsqueeze(0)))
        self.label_words = label_words

    @property
    def labels(self):
        if not hasattr(self, "_labels"):
            raise RuntimeError("Attribute `labels` is not set yet.")
        return self._labels

    @labels.setter
    def labels(self, labels):
        if labels is not None:
            self._labels = sorted(labels)

    @property
    def label_words(self):
        if not hasattr(self, "_label_words"):
            raise RuntimeError("Mapping from labels to words is not set yet.")
        return self._label_words

    @label_words.setter
    def label_words(self, label_words: Dict):
        if label_words is None:
            return None
        self.labels = list(label_words.keys())
        self.labels_to_ids = {label: idx for idx, label in enumerate(self._labels)}
        self._words = []
        for label in self._labels:
            words = label_words[label]
            if isinstance(words, str):
                words = [words]
            self._words.append(words)
        self._label_words = {label: word for label, word in zip(self._labels, self._words)}
        self.preprocess_label_words()
        self.create_parameters()

    @abstractmethod
    def create_parameters(self):
        """
        A hook to create parameters for mapping from labels to words.
        """
        raise NotImplementedError

    def preprocess_label_words(self):
        label_token_ids = []
        for label_word in self._words:
            word_token_ids = []
            for word in label_word:
                token_ids = self.tokenizer.encode(word, add_special_tokens=False, return_token_type_ids=False)
                word_token_ids.append(token_ids["input_ids"])
            label_token_ids.append(word_token_ids)

        max_num_words = max([len(words) for words in self._words])
        max_num_tokens = max(
            [max([len(token_ids) for token_ids in word_token_ids]) for word_token_ids in label_token_ids]
        )
        token_ids_shape = [len(self.labels), max_num_words, max_num_tokens]
        token_ids = np.zeros(token_ids_shape)
        word_mask = np.zeros(token_ids_shape[:-1])
        token_mask = np.zeros(token_ids_shape)
        for label_id, word_token_ids in enumerate(label_token_ids):
            word_mask[label_id][: len(word_token_ids)] = 1
            for word_id, tokens in enumerate(word_token_ids):
                token_ids[label_id][word_id][: len(tokens)] = tokens
                token_mask[label_id][word_id][: len(tokens)] = 1
        self.token_ids = paddle.to_tensor(token_ids, dtype="int64", stop_gradient=True)
        self.word_mask = paddle.to_tensor(word_mask, dtype="float32", stop_gradient=True)
        self.token_mask = paddle.to_tensor(token_mask, dtype="int64", stop_gradient=True)

    def convert_labels_to_ids(self, label: str):
        assert isinstance(label, str)
        return self.labels_to_ids[label]

    def convert_ids_to_labels(self, index: int):
        assert isinstance(index, int)
        return self.labels[index]

    def project(self, outputs):
        """
        Fetch label word predictions from outputs over vocabulary.
        """
        token_ids = self.token_ids.reshape([-1])
        label_token_outputs = outputs.index_select(index=token_ids, axis=-1)
        label_shape = [*outputs.shape[:-1], *self.token_ids.shape]
        label_token_outputs = label_token_outputs.reshape(label_shape)
        label_word_outputs = self.aggregate(label_token_outputs, self.token_mask, self.token_aggregate_type)
        label_word_outputs -= 1e4 * (1 - self.word_mask)
        return label_word_outputs

    def process_outputs(self, outputs, masked_positions: Tensor = None):
        """
        Process outputs of `PretrainedModelForMaskedLM` over vocabulary.
        """
        if masked_positions is None:
            return outputs
        batch_size, _, num_pred = outputs.shape
        outputs = outputs.reshape([-1, num_pred])
        outputs = paddle.gather(outputs, masked_positions)
        outputs = outputs.reshape([batch_size, -1, num_pred])
        return outputs

    def aggregate(self, outputs: Tensor, mask: Tensor, atype: str):
        """
        Aggregate multiple tokens/words for each word/label.
        """
        mask = mask.unsqueeze(0)
        if atype == "mean":
            outputs = outputs * mask
            outputs = outputs.sum(axis=-1) / (mask.sum(axis=-1) + 1e-15)
        elif atype == "max":
            outputs = (outputs - 1e4 * (1 - mask)).max(axis=-1)
        elif atype == "first":
            index = paddle.to_tensor([0])
            outputs = paddle.index_select(outputs, index, axis=-1)
        else:
            raise ValueError("Strategy {} is not supported to aggregate multiple " "tokens.".format(atype))
        return outputs

    def normalize(self, outputs: Tensor):
        """
        Normalize the outputs over the whole vocabulary.
        """
        batch_size = outputs.shape[0]
        outputs = F.softmax(outputs.reshape([batch_size, -1]), axis=-1).reshape(outputs.shape)
        return outputs

    def calibrate(self, label_word_outputs: Tensor):
        """
        Calibrate predictions with pre-defined weights over the whole vocabulary.
        """
        if self.label_token_weight.dim() != 1:
            raise ValueError("Weights of label tokens should be a 1-D tensor.")
        weight_shape = self.label_token_weight.shape
        output_shape = label_word_outputs.shape
        if weight_shape[1:] != output_shape[1:] or weight_shape[0] != 1:
            raise ValueError(
                "Shapes of label token weights and predictions do not match, "
                "got {} and {}.".format(weight_shape, output_shape)
            )
        label_word_outputs /= self.label_token_weight + 1e-15
        batch_size = label_word_outputs.shape0[0]
        label_word_outputs = paddle.mean(label_word_outputs.reshape([batch_size, -1])).reshape(output_shape)

        return label_word_outputs

    def save(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        verb_config_file = os.path.join(save_path, VERBALIZER_CONFIG_FILE)
        with open(verb_config_file, "w", encoding="utf-8") as fp:
            json.dump(self.label_words, fp, ensure_ascii=False)
        verb_params_file = os.path.join(save_path, VERBALIZER_PARAMETER_FILE)
        verb_state_dict = self.state_dict()
        if len(verb_state_dict) > 0:
            paddle.save(self.state_dict(), verb_params_file)

    @classmethod
    def load_from(cls, data_path: os.PathLike, tokenizer: PretrainedTokenizer):
        verb_config_file = os.path.join(data_path, VERBALIZER_CONFIG_FILE)
        if not os.path.isfile(verb_config_file):
            raise ValueError("{} not found under {}".format(VERBALIZER_CONFIG_FILE, data_path))
        with open(verb_config_file, "r") as fp:
            label_words = json.load(fp)

        verbalizer = cls(label_words, tokenizer)
        verb_state_file = os.path.join(data_path, VERBALIZER_PARAMETER_FILE)
        if os.path.isfile(verb_state_file):
            verbalizer.set_state_dict(paddle.load(verb_state_file))
            logger.info("Loading verbalizer state dict from {}".format(verb_state_file))
        return verbalizer


class ManualVerbalizer(Verbalizer):
    """
    ManualVerbalizer defines mapping from labels to words manually.

    Args:
        label_words (`dict`):
            Define the mapping from labels to a single or multiple words.
        tokenizer (`PretrainedTokenizer`):
            An instance of PretrainedTokenizer for label word tokenization.
    """

    def __init__(self, label_words: Dict, tokenizer: PretrainedTokenizer, **kwargs):
        super(ManualVerbalizer, self).__init__(label_words=label_words, tokenizer=tokenizer, **kwargs)

    def create_parameters(self):
        return None

    def aggregate_multiple_mask(self, outputs: Tensor, atype: str = None):
        if atype is None:
            return outputs
        assert outputs.ndim == 3
        if atype == "mean":
            outputs = outputs.sum(axis=1)
        elif atype == "max":
            outputs = outputs.max(axis=1)
        elif atype == "first":
            index = paddle.to_tensor([0])
            outputs = paddle.index_select(outputs, index, axis=1)
        elif atype == "product":
            new_outputs = outputs[:, 0, :]
            for index in range(1, outputs.shape[1]):
                new_outputs *= outputs[:, index, :]
            outputs = new_outputs
        else:
            raise ValueError("Strategy {} is not supported to aggregate multiple " "tokens.".format(atype))
        return outputs

    def process_outputs(self, outputs: Tensor, masked_positions: Tensor = None):
        """
        Process outputs over the vocabulary, including the following steps:

        (1) Project outputs into the outputs of corresponding word.

        If self.post_log_softmax is True:

            (2) Normalize over all label words.

            (3) Calibrate (optional)

        (4) Aggregate multiple words for each label.

        Args:
            outputs (`Tensor`):
                The outputs of `PretrainedModel` which class name ends with
                `ForMaskedLM`.
        Returns:
            The prediction outputs over labels (`Tensor`).
        """
        outputs = super(ManualVerbalizer, self).process_outputs(outputs, masked_positions)
        label_word_outputs = self.project(outputs)

        if self.post_log_softmax:
            label_word_outputs = self.normalize(label_word_outputs)

            if self.label_token_weight is not None:
                label_word_outputs = self.calibrate(label_word_outputs)

            label_word_outputs = paddle.log(label_word_outputs + 1e-15)

        label_outputs = self.aggregate(label_word_outputs, self.word_mask, self.word_aggregate_type)
        label_outputs = self.aggregate_multiple_mask(label_outputs, self.mask_aggregate_type)
        return label_outputs


class MaskedLMIdentity(nn.Layer):
    """
    Identity layer with the same arguments as the last linear layer in
    `PretrainedModel` whose name ends with `ForMaskedLM`.
    """

    def __init__(self):
        super(MaskedLMIdentity, self).__init__()

    def forward(self, sequence_output, masked_positions=None):
        return sequence_output


class SoftVerbalizer(Verbalizer):
    """
    SoftVerbalizer for the WARP method.

    Args:
        label_words (`dict`):
            Define the mapping from labels to a single or multiple words.
        tokenizer (`PretrainedTokenizer`):
            An instance of PretrainedTokenizer for label word tokenization.
        model (`PretrainedModel`):
            An instance of PretrainedModel with class name ends with `ForMaskedLM`
    """

    LAST_WEIGHT = ["ErnieForMaskedLM", "BertForMaskedLM"]
    LAST_LINEAR = ["AlbertForMaskedLM", "RobertaForMaskedLM"]

    def __init__(self, label_words: Dict, tokenizer: PretrainedTokenizer, model: PretrainedModel, **kwargs):
        super(SoftVerbalizer, self).__init__(label_words=label_words, tokenizer=tokenizer, model=model, **kwargs)
        del self.model
        setattr(model, self.head_name[0], MaskedLMIdentity())

    def create_parameters(self):
        # Only the first word used for initialization.
        if self.token_ids.shape[1] != 1:
            logger.warning("Only the first word for each label is used for" " initialization.")
            index = paddle.to_tensor([0])
            self.token_ids = paddle.index_select(self.token_ids, index, axis=1)
            self.token_mask = paddle.index_select(self.token_mask, index, axis=1)
            self.word_mask = paddle.ones([len(self.labels), 1])
        self._extract_head(self.model)

    def process_outputs(self, outputs: Tensor, masked_positions: Tensor = None):
        outputs = super(SoftVerbalizer, self).process_outputs(outputs, masked_positions)
        return self.head(outputs).squeeze(1)

    def head_parameters(self):
        if isinstance(self.head, nn.Linear):
            return [(n, p) for n, p in self.head.named_parameters()]
        else:
            return [(n, p) for n, p in self.head.named_parameters() if self.head_name[1] in n]

    def non_head_parameters(self):
        if isinstance(self.head, nn.Linear):
            return []
        else:
            return [(n, p) for n, p in self.head.named_parameters() if self.head_name[1] not in n]

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
                    setattr(self.head, name, nn.Linear(module.weight.shape[0], len(self.labels), bias_attr=False))
                    getattr(self.head, name).weight.set_value(self._create_init_weight(module.weight))
                    self.head_name.append(name)
                    break
        elif model_type in self.LAST_WEIGHT:
            last_name = [n for n, p in model.named_children()][-1]
            head = getattr(model, last_name)
            self.head_name = [last_name]
            # OnlyMLMHead
            if model_type in ["ErnieForMaskedLM", "BertForMaskedLM"]:
                last_name = [n for n, p in head.named_children()][-1]
                self.head = copy.deepcopy(getattr(head, last_name))
                self.head_name.append("decoder")
            else:
                self.head = copy.deepcopy(head)

            # LMPredictionHead
            module = paddle.to_tensor(getattr(self.head, "decoder_weight"))
            new_head = nn.Linear(len(self.labels), module.shape[1], bias_attr=False)
            new_head.weight.set_value(self._create_init_weight(module.T).T)
            setattr(self.head, "decoder_weight", new_head.weight)
            getattr(self.head, "decoder_weight").stop_gradient = False
            if hasattr(self.head, "decoder_bias"):
                setattr(
                    self.head,
                    "decoder_bias",
                    self.head.create_parameter(shape=[len(self.labels)], dtype=new_head.weight.dtype, is_bias=True),
                )
                getattr(self.head, "decoder_bias").stop_gradient = False
        else:
            raise NotImplementedError(
                f"Please open an issue to request for support of {model_type} or contribute to PaddleNLP."
            )

    def _create_init_weight(self, weight, is_bias=False):
        token_ids = self.token_ids.squeeze(1)
        token_mask = self.token_mask.squeeze(1)
        aggr_type = self.token_aggregate_type
        if is_bias:
            bias = paddle.index_select(weight, token_ids.reshape([-1]), axis=0).reshape(token_ids.shape)
            bias = self.aggregate(bias, token_mask, aggr_type)
            return bias
        else:
            word_shape = [weight.shape[0], *token_ids.shape]
            weight = paddle.index_select(weight, token_ids.reshape([-1]), axis=1).reshape(word_shape)
            weight = self.aggregate(weight, token_mask, aggr_type)
            return weight


class MaskedLMVerbalizer(Verbalizer):
    """
    MaskedLMVerbalizer defines mapping from labels to words manually and supports
    multiple masks corresponding to multiple tokens in words.

    Args:
        label_words (`dict`):
            Define the mapping from labels to a single word. Only the first word
            is used if multiple words are defined.
        tokenizer (`PretrainedTokenizer`):
            An instance of PretrainedTokenizer for label word tokenization.
    """

    def __init__(self, label_words: Dict, tokenizer: PretrainedTokenizer, **kwargs):
        super(MaskedLMVerbalizer, self).__init__(label_words=label_words, tokenizer=tokenizer, **kwargs)

    def create_parameters(self):
        return None

    def aggregate_multiple_mask(self, outputs: Tensor, atype: str = "product"):
        assert outputs.ndim == 3
        token_ids = self.token_ids[:, 0, :].T
        batch_size, num_token, num_pred = outputs.shape
        results = paddle.index_select(outputs[:, 0, :], token_ids[0], axis=1)
        if atype == "first":
            return results

        for index in range(1, num_token):
            sub_results = paddle.index_select(outputs[:, index, :], token_ids[index], axis=1)
            if atype in ("mean", "sum"):
                results += sub_results
            elif atype == "product":
                results *= sub_results
            elif atype == "max":
                results = paddle.stack([results, sub_results], axis=-1)
                results = results.max(axis=-1)
            else:
                raise ValueError("Strategy {} is not supported to aggregate multiple " "tokens.".format(atype))
        if atype == "mean":
            results = results / num_token
        return results

    def process_outputs(self, outputs: Tensor, masked_positions: Tensor = None):
        if masked_positions is None:
            return outputs

        batch_size, _, num_pred = outputs.shape
        outputs = outputs.reshape([-1, num_pred])
        outputs = paddle.gather(outputs, masked_positions)
        outputs = outputs.reshape([batch_size, -1, num_pred])
        return outputs
