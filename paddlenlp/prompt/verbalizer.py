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

from paddlenlp.layers import Linear as TransposedLinear
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
        self.post_log_softmax = kwargs.get("post_log_softmax", True)
        self.label_token_weight = kwargs.get("label_token_weight", None)
        self.label_words = label_words
        if self.label_token_weight is not None:
            self.label_token_weight = self.normalize(self.project(self.label_token_weight.unsqueeze(0)))

    @property
    def labels(self):
        if not hasattr(self, "_labels"):
            raise RuntimeError("Attribute `labels` is not set yet.")
        return self._labels

    @labels.setter
    def labels(self, labels):
        raise NotImplementedError("Please use `label_words` to change `labels`.")

    @property
    def label_words(self):
        if not hasattr(self, "_label_words"):
            raise RuntimeError("Mapping from labels to words is not set yet.")
        return self._label_words

    @label_words.setter
    def label_words(self, label_words: Dict):
        if label_words is None:
            return None
        self._labels = sorted(list(label_words.keys()))
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
        self.word_mask = paddle.to_tensor(word_mask, dtype="int64", stop_gradient=True)
        self.token_mask = paddle.to_tensor(token_mask, dtype="int64", stop_gradient=True)

    def convert_labels_to_ids(self, label: str):
        assert isinstance(label, str)
        return self.labels_to_ids[label]

    def convert_ids_to_labels(self, index: int):
        assert isinstance(index, int)
        return self.labels[index]

    def project(self, outputs: Tensor):
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

    def process_outputs(self, outputs: Tensor, masked_positions: Tensor = None):
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
        if atype == "mean":
            outputs = outputs * mask
            outputs = outputs.sum(axis=-1) / (mask.sum(axis=-1) + 1e-15)
        elif atype == "max":
            outputs = (outputs - 1e4 * (1 - mask)).max(axis=-1)
        elif atype == "first":
            index = paddle.to_tensor([0])
            outputs = paddle.index_select(outputs, index, axis=-1).squeeze(axis=-1)
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

    def save(self, save_path: str):
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
            outputs = outputs.mean(axis=1)
        elif atype == "max":
            outputs = outputs.max(axis=1)
        elif atype == "first":
            index = paddle.to_tensor([0])
            outputs = paddle.index_select(outputs, index, axis=1).squeeze(1)
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
        # possible head parameters: decoder.weight, decoder_bias, bias
        return [(n, p) for n, p in self.head.named_parameters() if self.head_name[-1] in n or n == "bias"]

    def non_head_parameters(self):
        return [(n, p) for n, p in self.head.named_parameters() if self.head_name[-1] not in n and n != "bias"]

    def _extract_head(self, model: PretrainedModel):
        # Find the nn.Linear layer with in_features = vocab_size
        module_name = None
        for i in model.named_sublayers():
            if isinstance(i[1], TransposedLinear):
                module_name = i[0]
                break
        if module_name is None:
            raise ValueError("Can not find output layer, make sure type of the input model is AutoModelForMaskedLM.")

        # recursively get the parent module to the decoder linear layer
        parent_module = model
        attribute_chain = module_name.split(".")
        for name in attribute_chain[:-1]:
            parent_module = getattr(parent_module, name)
        self.head = copy.deepcopy(parent_module)

        # replace the decoder linear layer with a linear linear with the trimmed vocab size
        # we create a new decoder linear here instead of `resize_token_embeddings` because we only want to change the output embeddings
        # this also invalidates any previous tie_weights
        self.head_name = attribute_chain
        module_name = attribute_chain[-1]
        module = getattr(self.head, module_name)
        # modify weight
        module_weight = module.weight
        module_bias = module.bias
        selected_weight = self._create_init_weight(module_weight)
        selected_bias = self._create_init_weight(module_bias, is_bias=True)
        setattr(
            self.head, module_name, TransposedLinear(in_features=module.weight.shape[1], out_features=len(self.labels))
        )
        getattr(self.head, module_name).weight.set_value(selected_weight.T)
        getattr(self.head, module_name).bias.set_value(selected_bias)

    def _create_init_weight(self, weight: Tensor, is_bias: bool = False):
        token_ids = self.token_ids.squeeze(1)
        token_mask = self.token_mask.squeeze(1)
        aggr_type = self.token_aggregate_type
        if is_bias:
            bias = paddle.index_select(weight, token_ids.reshape([-1]), axis=0).reshape(token_ids.shape)
            bias = self.aggregate(bias, token_mask, aggr_type)
            return bias
        else:
            word_shape = [weight.shape[1], *token_ids.shape]
            weight = paddle.index_select(weight, token_ids.reshape([-1]), axis=0).reshape(word_shape)
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
        label_words = self.check_label_words_constraint(label_words)
        super(MaskedLMVerbalizer, self).__init__(label_words=label_words, tokenizer=tokenizer, **kwargs)

    def create_parameters(self):
        return None

    def check_label_words_constraint(self, label_words: Dict):
        assert isinstance(label_words, dict), "`label_words` mapping should be a dictionary."
        std_label_words = {}
        for label, word in label_words.items():
            if isinstance(word, str):
                word = [word]
            if len(word) > 1:
                word = word[:1]
                logger.info(f"More than one word for label `{label}`, only `{word[0]}` used.")
            std_label_words[label] = word
        word_length = [len(w[0]) for l, w in std_label_words.items()]
        if len(set(word_length)) > 1:
            raise ValueError(f"Length of all words for labels should be equal, but received {std_label_words}.")
        return std_label_words

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
                raise ValueError("Strategy {} is not supported to aggregate multiple tokens.".format(atype))
        if atype == "mean":
            results = results / num_token
        return results
