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

This module provide prompt definition methods.
"""

import json
import os
import re
import traceback
from abc import abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import paddle
import paddle.nn as nn
from paddle import Tensor

from paddlenlp.transformers import PretrainedModel, PretrainedTokenizer
from paddlenlp.utils.log import logger

from .prompt_tokenizer import MLMPromptTokenizer
from .prompt_utils import (
    masked_lm_forward_with_past_key_values,
    sequence_classification_forward_with_past_key_values,
)

__all__ = ["Template", "ManualTemplate", "SoftTemplate", "PrefixTemplate", "AutoTemplate", "UTCTemplate"]

# Template used to be saved in a file.
TEMPLATE_CONFIG_FILE = "template_config.json"
TEMPLATE_PARAMETER_FILE = "template_state.pdparams"

# Default values for some template attributes.
DEFAULT_MAX_OPTIONS = 10


class Template(nn.Layer):
    """
    Base class for [`Template`].

    Args:
        prompt (`str`):
            A template string which defines how to combine text and prompt.
        tokenizer (`PretrainedTokenizer`):
            An instance of PretrainedTokenizer used for tokenization.
        max_length (`int`):
            If set to a number, it will limit the total sequence returned so
            that it has a maximum length, including prompts.
    """

    template_special_tokens = ["text", "hard", "soft", "soft_id", "prefix", "sep", "mask", "options"]
    template_attributes = [
        "length",
        "encoder",
        "position",
        "token_type",
        "hidden_size",
        "add_omask",
        "add_prompt",
        "add_space",
        "truncate",
    ]
    input_feature_names = ["do_truncate", "token_types", "positions"]
    opt_token = "[OPT]"
    omask_token = "[O-MASK]"

    def __init__(self, prompt: str, tokenizer: PretrainedTokenizer, max_length: int, **kwargs):
        super(Template, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.tokenizer = tokenizer
        self.prompt_tokenizer = MLMPromptTokenizer(tokenizer, max_length)
        self.set_prompt(prompt)

    @property
    def prompt(self):
        return self._prompt

    @prompt.setter
    def prompt(self, prompt):
        logger.warning("Prompt can not be modified once set.")

    def set_prompt(self, prompt: str):
        if prompt is not None:
            if isinstance(prompt, str):
                self._prompt = self.parse_template_string(prompt)
            else:
                self._prompt = prompt
            self.do_truncate = self.create_truncation_sequence_from_prompt()
            self._check_template_special_tokens()
            self.example_keys = self.create_example_keys_from_prompt()
            self.token_types = self.create_token_type_sequence_from_prompt()
            self.positions = self.create_position_sequence_from_prompt()
            self.create_prompt_parameters()

    @abstractmethod
    def create_prompt_parameters(self):
        raise NotImplementedError

    def _check_template_special_tokens(self):
        valid_attr = self.template_special_tokens + self.template_attributes
        prompt_attr = []
        for part in self._prompt:
            prompt_attr.extend(list(part.keys()))
            if "add_prompt" in part:
                opt_prompt = part["add_prompt"]
                if self.opt_token not in opt_prompt:
                    raise ValueError("'{}' not found in option prompt.".format(self.opt_token))
            if "add_omask" in part:
                self._check_omask_token()
        diff_attr = set(prompt_attr) - set(valid_attr)
        if len(diff_attr) > 0:
            raise ValueError("Invalid attributes found in template: {}.".format(diff_attr))
        return True

    def _check_example_name(self, name: str, example: Dict[str, Any]):
        if name not in example:
            raise ValueError(
                "Unexpected value in template. Can not find keyword {} in example: {}".format(name, example)
            )
        return True

    def _check_omask_token(self):
        omask_example = """
        Add '[O-MASK]' to tokenizer to use `add_omask`.

        Examples:

        ```python
        omask_dict = {"additional_special_tokens": ["[O-MASK]"]}
        tokenizer.add_special_tokens(omask_dict)
        model.resize_token_embeddings(len(tokenizer))
        ```"""
        if self.omask_token not in self.tokenizer.additional_special_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.omask_token]})
            return True
            raise ValueError("'{}' not found in tokenizer.".format(self.omask_token) + omask_example)
        return True

    def build_inputs_with_prompt(
        self, example: Dict[str, Any], prompt: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Build input text sequences according to both prompt and example.

        Args:
            example (`Dict[str, Any]`):
                A data sample with corresponding keys as `prompt`.
            prompt (`Optional[List[Dict[str, Any]]]`):
                A sequence of dictionary which defines positions of prompt,
                input text and special tokens.
        """
        inputs = self._prompt.copy() if prompt is None else prompt.copy()

        for index, part in enumerate(inputs):
            if "text" in part:
                self._check_example_name(part["text"], example)
                inputs[index] = str(example[part["text"]])
            elif "mask" in part:
                if "length" not in part:
                    part["length"] = 1
                inputs[index] = self.tokenizer.mask_token * part["length"]
            elif "sep" in part:
                inputs[index] = self.tokenizer.sep_token
            elif "hard" in part:
                inputs[index] = part["hard"]
            elif "options" in part:
                if not isinstance(part["options"], list):
                    self._check_example_name(part["options"], example)
                    labels = example[part["options"]]
                    labels = [labels] if isinstance(labels, str) else labels
                else:
                    labels = part["options"]
                if "add_prompt" in part:
                    opt_prompt = part["add_prompt"]
                    labels = [opt_prompt.replace(self.opt_token, x) for x in labels]
                if "add_omask" in part:
                    labels = [self.omask_token + x for x in labels]
                inputs[index] = "".join(labels)
            else:
                inputs[index] = part

            if "add_space" in part:
                inputs[index] = " " + inputs[index]
        return inputs

    def create_token_type_sequence_from_prompt(self, prompt: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        prompt = self._prompt if prompt is None else prompt
        last_token_type = 0
        token_type_ids = []
        for part in prompt:
            if "token_type" in part:
                last_token_type = part["token_type"]
            token_type_ids.append(last_token_type)
        return token_type_ids

    def create_position_sequence_from_prompt(self, prompt: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        prompt = self._prompt if prompt is None else prompt
        position_ids = []
        for part in prompt:
            if "position" in part:
                position_ids.append(part["position"])
            else:
                position_ids.append(-1)
        return position_ids

    def create_truncation_sequence_from_prompt(self, prompt: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        prompt = self._prompt.copy() if prompt is None else prompt.copy()
        do_truncate = []
        for part in prompt:
            if "truncate" in part:
                do_truncate.append(part["truncate"])
            elif "text" in part:
                do_truncate.append(True)
            else:
                do_truncate.append(False)
        return do_truncate

    def create_example_keys_from_prompt(self):
        example_keys = set()
        for part in self.prompt:
            if "text" in part:
                example_keys.add(part["text"])
            if "options" in part and isinstance(part["options"], list):
                example_keys.update(set(part["options"]))
        if len(example_keys) == 0:
            raise ValueError('No `text` keyword in template: "{}", please check it again.'.format(self.prompt))
        return example_keys

    def encode(self, example: Dict[str, Any]):
        input_text = self.build_inputs_with_prompt(example)
        input_names, input_values = ["text"], [input_text]
        for name in self.input_feature_names:
            input_names.append(name)
            input_values.append(getattr(self, name, None))

        inputs = []
        for value in list(zip(*input_values)):
            inputs.append(dict(zip(input_names, value)))

        input_dict = self.prompt_tokenizer(inputs)
        unused_example = {k: v for k, v in example.items() if k not in self.example_keys}

        return {**input_dict, **unused_example}

    def __call__(self, example: Dict[str, Any]):
        return self.encode(example=example)

    @abstractmethod
    def process_batch(self, input_dict):
        raise NotImplementedError

    def save(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        template_config_file = os.path.join(save_path, TEMPLATE_CONFIG_FILE)
        template_class = self.__class__.__name__
        with open(template_config_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(self._prompt, ensure_ascii=False) + "\n")
            fp.write(json.dumps({"class": template_class}, ensure_ascii=False) + "\n")
        template_param_file = os.path.join(save_path, TEMPLATE_PARAMETER_FILE)
        template_state_dict = self.state_dict()
        if len(template_state_dict) > 0:
            paddle.save(template_state_dict, template_param_file)

    @staticmethod
    def extract_template_keywords(prompt: List[Dict[str, Any]]):
        keywords = set()
        for part in prompt:
            keywords.update(part.keys())
        return keywords

    @staticmethod
    def parse_template_string(prompt: str, left_token: Optional[str] = "{", right_token: Optional[str] = "}"):
        """
        Parse the defined string as a sequence of dictionaries.

        Args:
            prompt: A string comprised of nestable {}, [], integers and strings.

        Returns:
            A list of dictionaries corresponding to the input string.

            For example, if we define `prompt` as

            "{'text': 'hypothesis'}基于这一假设{'mask'}推断出{'options': 'label.txt'}",

            then this function returns

            [{"text": "hypothesis"}, {"hard": "基于这一假设"}, {"mask": null},
             {"hard": "推断出"}, {"options": ["正确", "错误"]}].

        Raises:
            ValueError: A error occurred parsing an string with unmatched punctuations.
        """
        left_stack = []
        parsed = []
        index = 0
        while index < len(prompt):
            # Delete extra spaces.
            part = {"add_space": " "} if prompt[index] == " " else {}
            while index < len(prompt) and prompt[index] == " ":
                index += 1
            if index == len(prompt):
                break
            # Parse blocks with paired tokens like "{ }".
            if prompt[index] == left_token:
                left_index = index
                while index < len(prompt):
                    if prompt[index] == left_token:
                        left_stack.append(index)
                    elif prompt[index] == right_token:
                        left_stack.pop()
                        if len(left_stack) == 0:
                            break
                    index += 1
                if index == len(prompt) and len(left_stack) > 0:
                    raise ValueError(
                        "{} at position {} has no corresponding {}".format(left_token, left_index, right_token)
                    )
                try:
                    part_dict = eval(prompt[left_index : index + 1])
                    if isinstance(part_dict, set):
                        part_dict = {k: None for k in part_dict}
                    part.update(part_dict)
                except SyntaxError:
                    logger.error(traceback.format_exc())
                    exit()
                index += 1
            # Parse simplified discrete prompts.
            else:
                left_index = index
                while index < len(prompt) and prompt[index] != left_token:
                    index += 1
                part["hard"] = prompt[left_index:index].rstrip(" ")

            if "options" in part:
                if os.path.isfile(part["options"]):
                    with open(part["options"], "r") as fp:
                        labels = [x.strip() for x in fp]
                    part["options"] = labels
                    part["length"] = len(labels)
                elif "length" not in "options":
                    part["length"] = DEFAULT_MAX_OPTIONS
            if "length" in part:
                assert part["length"] > 0
                if "hard" in part:
                    logger.warning("Ignore `length` attribute for keyword `hard`.")
            if "position" in part:
                assert part["position"] >= 0
            if "token_type" in part:
                assert part["token_type"] in (0, 1)
            parsed.append(part)
        return parsed


class ManualTemplate(Template):
    """
    ManualTemplate for discrete prompt methods, such as PET, EFL.

    Args:
        prompt (`str`):
            A template string which defines how to combine text and prompt.
        tokenizer (`PretrainedTokenizer`):
            An instance of PretrainedTokenizer used for tokenization.
        max_length (`int`):
            If set to a number, it will limit the total sequence returned so
            that it has a maximum length, including prompts.
    """

    template_special_tokens = ["text", "hard", "sep", "mask", "options"]
    template_attributes = ["length", "position", "token_type", "add_prompt", "add_space", "add_omask", "truncate"]

    def __init__(self, prompt: str, tokenizer: PretrainedTokenizer, max_length: int):
        super(ManualTemplate, self).__init__(prompt, tokenizer, max_length)

    def create_prompt_parameters(self):
        return None

    def process_batch(self, input_dict):
        return input_dict


class SoftLSTM(nn.Layer):
    """
    LSTM encoder for soft token embeddings.
    """

    def __init__(self, input_size, hidden_size, output_size, activation):
        super(SoftLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=2, direction="bidirect", time_major=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size), activation, nn.Linear(hidden_size, output_size)
        )

    def forward(self, embeds):
        hidden_states, _ = self.lstm(embeds)
        return self.mlp(hidden_states)


class SoftTemplate(Template):
    """
    SoftTemplate for continuous prompt methods on the input layer.

    Args:
        prompt (`str`):
            A template string which defines how to combine text and prompt.
        tokenizer (`PretrainedTokenizer`):
            An instance of PretrainedTokenizer used for tokenization.
        max_length (`int`):
            If set to a number, it will limit the total sequence returned so
            that it has a maximum length, including prompts.
        word_embeddings (`Tensor`):
            The word embeddings of pretrained models, which can be obtained by
            calling `model.get_input_embeddings().weight`.
        soft_embeddings (`Tensor`):
            The embeddings of soft tokens, which overwrites `word_embeddings`
            as initial weights when defined.
    """

    template_special_tokens = ["text", "hard", "soft", "soft_id", "sep", "mask", "options"]
    input_feature_names = ["do_truncate", "token_types", "positions", "soft_tokens", "encoder_ids"]

    def __init__(
        self,
        prompt: str,
        tokenizer: PretrainedTokenizer,
        max_length: int,
        word_embeddings: Tensor,
        soft_embeddings: Tensor = None,
    ):
        super(SoftTemplate, self).__init__(
            prompt, tokenizer, max_length, word_embeddings=word_embeddings, soft_embeddings=soft_embeddings
        )

    def named_parameters(self):
        named_params = [(n, p) for n, p in self.soft_embeddings.named_parameters()]
        named_params.extend([(n, p) for n, p in self.encoder_list.named_parameters()])
        return named_params

    def parameters(self):
        return [p for n, p in self.named_parameters()]

    def create_prompt_parameters(self):
        self._prompt, soft_token_config = self.parse_soft_prompt()
        self.embed_size = self.word_embeddings.weight.shape[1]
        soft2word, self.soft_tokens, self.num_soft_token = soft_token_config
        self._init_soft_parameters(soft2word)
        self.encoder_ids, self.encoder_list = self._create_soft_encoders()

    def process_batch(self, input_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Convert input_ids to inputs_embeds.

        Soft tokens are encoded soft_embeddings with predefined encoders.
        For other tokens, use word embeddings in pretrained model.
        """
        word_embeds = self.word_embeddings(input_dict["input_ids"])
        if "attention_mask" not in input_dict or input_dict["attention_mask"] is None:
            pad_token_id = self.tokenizer.pad_token_id
            attention_mask = paddle.unsqueeze(
                (input_dict["input_ids"] == pad_token_id).astype("float32") * -1e4, axis=[1, 2]
            )
            input_dict["attention_mask"] = attention_mask
        input_dict["input_ids"] = None
        soft_embeds = self.soft_embeddings(input_dict["soft_token_ids"])
        soft_shape = soft_embeds.shape
        soft_embeds = soft_embeds.reshape([-1, soft_shape[-1]])
        for encoder_id in range(1, len(self.encoder_list)):
            to_encode = paddle.where(input_dict["encoder_ids"] == encoder_id)
            to_encode = to_encode[0] * soft_shape[1] + to_encode[1]
            to_encode = to_encode.squeeze(1)
            to_encode_embeds = soft_embeds[to_encode]
            to_encode_embeds = to_encode_embeds.reshape([soft_shape[0], -1, soft_shape[-1]])
            encoder = self.encoder_list[encoder_id]
            encoded = encoder(to_encode_embeds)
            encoded = encoded.reshape([-1, soft_shape[-1]])
            soft_embeds = paddle.scatter(soft_embeds, to_encode, encoded)
        soft_embeds = soft_embeds.reshape([soft_shape[0], -1, soft_shape[-1]])
        soft_token_ids = input_dict["soft_token_ids"].unsqueeze(2)
        input_dict["inputs_embeds"] = paddle.where(soft_token_ids > 0, soft_embeds, word_embeds)
        return input_dict

    def parse_soft_prompt(self):
        """
        Unify the form of continuous prompts as {"soft": "xxx"} and create
        continuous token id sequence for each part in template.

        Returns:
            `List[Dict[str, str]]`: Template with continuous prompt formated as {"soft": "xxx"}.
            `Tuple[Dict[int, int], List[List[int]], int]`:
                - Mapping from continuous ids to word ids for initialization.
                - Continuous ids for each part. Id 0 denotes none-continuous part.
                - Number of unique continuous tokens.
        """
        prompt = self._prompt.copy()
        num_soft_token = 1
        soft_prompt = []
        soft_token_ids = []
        soft2word = {}
        soft_id_reindex = {}

        for part in prompt:
            part_prompt = None
            # Copy non-continuous prompt part.
            if "soft" not in part and "soft_id" not in part:
                soft_prompt.append(part)
                soft_token_ids.append(None)

            # Deal with continuous prompt with specific initialization.
            elif "soft" in part and part["soft"] is not None:

                # Get word tokens for initialization.
                if "add_space" in part:
                    part["soft"] = part["add_space"] + part["soft"]
                word_token_ids = self.tokenizer(part["soft"], add_special_tokens=False, return_token_type_ids=False)[
                    "input_ids"
                ]

                # Create continuous token ids.
                soft_id_list = list(range(num_soft_token, num_soft_token + len(word_token_ids)))
                num_soft_token += len(word_token_ids)

                for soft_id, word_id in zip(soft_id_list, word_token_ids):
                    soft2word[soft_id] = word_id

                # Check `length` if exists.
                if "length" in part:
                    if part["length"] < len(word_token_ids):
                        logger.warning("Ignore `length` because it is less than the length of defined word sequence.")
                    elif part["length"] > len(word_token_ids):
                        length = part["length"] - len(word_token_ids)
                        soft_id_list += list(range(num_soft_token, num_soft_token + length))
                        num_soft_token += length
                        part["soft"] += self.tokenizer.unk_token * length

                soft_token_ids.append(soft_id_list)
                part_prompt = {"soft": part["soft"]}

                # Check or record `soft_id` if exists.
                if "soft_id" in part:
                    if part["soft_id"] in soft_id_reindex:
                        assert soft_id_list == soft_id_reindex[part["soft_id"]]
                    else:
                        soft_id_reindex[part["soft_id"]] = soft_id_list

            # Deal with continuous prompt defined by `soft_id`.
            elif "soft_id" in part and part["soft_id"] in soft_id_reindex:
                soft_id_list = soft_id_reindex[part["soft_id"]]
                if "length" in part:
                    logger.warning("Ignore `length` because it is incompatible with existing `soft_id`.")
                soft_token_ids.append(soft_id_list)
                part_prompt = {"soft": [self.tokenizer.unk_token] * len(soft_id_list)}

            # Deal with continuous prompt with random initialization.
            else:
                if "length" not in part:
                    part["length"] = 1
                soft_id_list = list(range(num_soft_token, num_soft_token + part["length"]))
                num_soft_token += part["length"]
                soft_token_ids.append(soft_id_list)
                if "soft_id" in part:
                    soft_id_reindex[part["soft_id"]] = soft_id_list
                part_prompt = {"soft": [self.tokenizer.unk_token] * len(soft_id_list)}
            if part_prompt is not None:
                for key in part:
                    if key not in ["soft", "soft_id", "length", "add_space"]:
                        part_prompt[key] = part[key]
                soft_prompt.append(part_prompt)

        if num_soft_token == 1:
            raise ValueError("Soft prompt expected for SoftTemplate, but get {}.".format(self._prompt))

        soft_token_config = (soft2word, soft_token_ids, num_soft_token)

        return soft_prompt, soft_token_config

    def _init_soft_parameters(self, soft2word: Dict[int, int]):
        if self.soft_embeddings is not None:
            if self.soft_embeddings.weight.shape[0] != self.num_soft_token:
                raise ValueError(
                    "Given soft embeddings are incompatible with those "
                    'defined in template "{}"'.format(self._prompt)
                )
        else:
            self.soft_embeddings = nn.Embedding(self.num_soft_token, self.embed_size)
            weight = self.soft_embeddings.weight.clone().detach()
            for soft_id, word_id in soft2word.items():
                # squeeze() is used here to be backward compatible with 0-D tensor introduced in paddle 2.5
                word_id = paddle.to_tensor(word_id).squeeze()
                weight[soft_id] = self.word_embeddings(word_id)
            self.soft_embeddings.weight.set_value(weight)

    def _create_soft_encoders(self, output_size: int = None, activation: nn.Layer = None):
        encoder_list = [nn.Identity()]
        encoder2id = {}
        encoder_ids = []
        output_size = self.embed_size if output_size is None else output_size
        activation = nn.ReLU() if activation is None else activation
        for part in self._prompt:
            if "encoder" not in part or part["encoder"] is None:
                encoder_ids.append(0)
            else:
                if part["encoder"] not in encoder2id:
                    encoder2id[part["encoder"]] = len(encoder_list)
                    encoder_ids.append(len(encoder_list))
                    if "hidden_size" in part:
                        hidden_size = part["hidden_size"]
                    else:
                        hidden_size = self.embed_size
                    if part["encoder"] == "lstm":
                        encoder_list.append(SoftLSTM(self.embed_size, hidden_size, output_size, activation))
                    elif part["encoder"] == "mlp":
                        encoder_list.append(
                            nn.Sequential(
                                nn.Linear(self.embed_size, hidden_size),
                                activation,
                                nn.Linear(hidden_size, output_size),
                            )
                        )
                    else:
                        raise ValueError("Encoder {} not supported.".format(part["encoder"]))
                else:
                    encoder_ids.append(encoder2id[part["encoder"]])
        encoder_list = nn.LayerList(encoder_list)
        return encoder_ids, encoder_list

    def build_inputs_with_prompt(
        self, example: Dict[str, Any], prompt: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        inputs = super(SoftTemplate, self).build_inputs_with_prompt(example, prompt)
        for index, part in enumerate(inputs):
            if isinstance(part, dict) and "soft" in part:
                inputs[index] = part["soft"]
        return inputs

    def save(self, save_path):
        super(SoftTemplate, self).save(save_path)
        template_param_file = os.path.join(save_path, TEMPLATE_PARAMETER_FILE)
        paddle.save(self.state_dict(), template_param_file)


class PrefixTemplate(SoftTemplate):
    """
    PrefixTemplate for continuous prompt methods on every layer.

    Args:
        prompt (`str`):
            A template string which defines how to combine text and prompt.
        tokenizer (`PretrainedTokenizer`):
            An instance of PretrainedTokenizer used for tokenization.
        max_length (`int`):
            If set to a number, it will limit the total sequence returned so
            that it has a maximum length, including prompts.
        model (`PretrainedModel`):
            An instance of PretrainedModel.
    """

    template_special_tokens = ["text", "hard", "prefix", "soft", "sep", "mask", "options"]
    input_feature_names = ["do_truncate", "token_types", "positions", "soft_tokens", "encoder_ids"]

    def __init__(
        self,
        prompt: str,
        tokenizer: PretrainedTokenizer,
        max_length: int,
        model: PretrainedModel,
        prefix_dropout: float = 0.1,
    ):
        self.n_layer, self.n_heads = self._get_config(model)
        super(PrefixTemplate, self).__init__(prompt, tokenizer, max_length, model.get_input_embeddings())
        self.dropout = nn.Dropout(p=prefix_dropout)

    @staticmethod
    def _get_config(model):
        names = [n for n, p in model.named_parameters() if "layers" in n]
        pattern = re.compile(r".*?\.(\d+)\..*?")
        indices = []
        for name in names:
            result = pattern.match(name)
            if result is not None:
                indices.append(int(result.group(1)))
        num_layer = max(indices) + 1
        layer_names = names[0].split(".")[:-2]
        layer = model
        for name in layer_names:
            layer = getattr(layer, name)
        num_heads = layer.num_heads

        return num_layer, num_heads

    def parse_soft_prompt(self):
        prompt = self._prompt.copy()

        for index, part in enumerate(prompt):
            if "soft" in part:
                raise ValueError("Keyward `soft` should not be used in PrefixTemplate.")
            if "prefix" not in part:
                continue
            if index != 0:
                raise ValueError("Keyword `prefix` should locate at the beginning of template.")
            part["soft"] = part["prefix"]
            part.pop("prefix")
            if "encoder" not in part:
                part["encoder"] = "mlp"
            prompt[index] = part

        self._prompt = prompt
        return super(PrefixTemplate, self).parse_soft_prompt()

    def process_model(self, model):
        if model.__class__.__name__.endswith("ForSequenceClassification"):
            model.forward = partial(sequence_classification_forward_with_past_key_values, self=model)
        elif model.__class__.__name__.endswith("ForMaskedLM"):
            model.forward = partial(masked_lm_forward_with_past_key_values, self=model)
        return model

    def process_batch(self, input_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        word_embeds = self.word_embeddings(input_dict["input_ids"])
        batch_size, _ = input_dict["soft_token_ids"].shape

        soft_token_ids = paddle.masked_select(input_dict["soft_token_ids"], input_dict["soft_token_ids"] > 0)
        soft_token_ids = soft_token_ids.reshape([batch_size, -1])
        _, soft_len = soft_token_ids.shape

        token_type_ids = paddle.masked_select(input_dict["token_type_ids"], input_dict["soft_token_ids"] == 0)
        input_dict["token_type_ids"] = token_type_ids.reshape([batch_size, -1])
        position_ids = paddle.masked_select(input_dict["position_ids"], input_dict["soft_token_ids"] == 0)
        input_dict["position_ids"] = position_ids.reshape([batch_size, -1])
        if "masked_position" in input_dict and input_dict["masked_positions"] is not None:
            input_dict["masked_positions"] = input_dict["masked_positions"] - soft_len
        input_dict["inputs_embeds"] = paddle.concat(
            [word_embeds[:, 0, :].unsqueeze(1), word_embeds[:, soft_len + 1 :, :]], axis=1
        )

        if "attention_mask" not in input_dict or input_dict["attention_mask"] is None:
            pad_token_id = self.tokenizer.pad_token_id
            attention_mask = paddle.unsqueeze(
                (input_dict["input_ids"] == pad_token_id).astype("float32") * -1e4, axis=[1, 2]
            )
            input_dict["attention_mask"] = attention_mask
        input_dict["input_ids"] = None
        input_dict.pop("soft_token_ids")
        input_dict.pop("encoder_ids")

        soft_embeds = self.soft_embeddings(soft_token_ids)
        soft_embeds = self.encoder_list[1](soft_embeds)
        soft_embeds = soft_embeds.reshape(
            [batch_size, soft_len, self.n_layer * 2, self.n_heads, self.embed_size // self.n_heads]
        )

        soft_embeds = self.dropout(soft_embeds)
        soft_embeds = paddle.transpose(soft_embeds, perm=[2, 0, 3, 1, 4])
        soft_embeds = paddle.split(soft_embeds, num_or_sections=self.n_layer)
        soft_embeds = [paddle.split(emb, 2) for emb in soft_embeds]
        soft_embeds = [[x.squeeze(0) for x in emb] for emb in soft_embeds]
        input_dict["past_key_values"] = tuple([tuple(emb) for emb in soft_embeds])
        return input_dict

    def _create_soft_encoders(self):
        output_size = self.embed_size * self.n_layer * 2
        activation = nn.Tanh()
        return super(PrefixTemplate, self)._create_soft_encoders(output_size, activation)


class AutoTemplate(object):
    """
    AutoTemplate can help you automatically create the relevant Template
    given the provided prompt.
    """

    default_text_keyword = "text_a"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            "{} is designed to be instantiated using {}.create_from("
            "prompt, tokenizer, max_length, ...)".format(self.__class__.__name__, self.__class__.__name__)
        )

    @classmethod
    def create_from(
        cls,
        prompt: str,
        tokenizer: PretrainedTokenizer,
        max_length: int = 512,
        model: PretrainedModel = None,
        soft_embeddings: Tensor = None,
        prefix_dropout: float = 0.1,
        template_class: str = None,
    ):
        # Default template if not defined.
        if prompt is None:
            prompt = "{'soft'}{'text': 'text_a'}{'mask'}"

        if isinstance(prompt, str):
            prompt = Template.parse_template_string(prompt)
        template_keywords = Template.extract_template_keywords(prompt)

        # Complement simplified template as ManualTemplate-style in form.
        if "text" not in template_keywords:
            prompt = [{"text": cls.default_text_keyword}] + prompt
            if "mask" not in template_keywords:
                prompt = prompt + [{"mask": None}]

        if template_class is None:
            if "prefix" in template_keywords:
                template_class = "PrefixTemplate"
            elif "soft" in template_keywords or "soft_id" in template_keywords:
                template_class = "SoftTemplate"
            else:
                template_class = "ManualTemplate"

        # Choose Template according to template keywords.
        if template_class == "PrefixTemplate":
            return PrefixTemplate(
                prompt=prompt, tokenizer=tokenizer, max_length=max_length, model=model, prefix_dropout=prefix_dropout
            )
        elif template_class == "SoftTemplate":
            word_embeddings = model.get_input_embeddings()
            return SoftTemplate(
                prompt=prompt,
                tokenizer=tokenizer,
                max_length=max_length,
                word_embeddings=word_embeddings,
                soft_embeddings=soft_embeddings,
            )
        elif template_class == "UTCTemplate":
            return UTCTemplate(tokenizer=tokenizer, max_length=max_length)
        elif template_class == "ManualTemplate":
            return ManualTemplate(prompt=prompt, tokenizer=tokenizer, max_length=max_length)
        else:
            raise ValueError(f"Unknown template: {template_class}.")

    @classmethod
    def load_from(
        cls, data_path: os.PathLike, tokenizer: PretrainedTokenizer, max_length: int, model: PretrainedModel = None
    ):
        template_config_file = os.path.join(data_path, TEMPLATE_CONFIG_FILE)
        if not os.path.isfile(template_config_file):
            raise ValueError("{} not found under {}".format(TEMPLATE_CONFIG_FILE, data_path))
        with open(template_config_file, "r") as fp:
            config = [x.strip() for x in fp]
            prompt = json.loads(config[0])
            if len(config) > 1:
                template_class = json.loads(config[1])["class"]
            else:
                template_class = None  # Compatible with previous versions
        template = cls.create_from(
            prompt=prompt, tokenizer=tokenizer, max_length=max_length, model=model, template_class=template_class
        )
        template_param_file = os.path.join(data_path, TEMPLATE_PARAMETER_FILE)
        if os.path.isfile(template_param_file):
            template.set_state_dict(paddle.load(template_param_file))
        return template


class UTCTemplate(Template):
    """
    Template for Unified Tag Classification.
    """

    template_special_tokens = ["text", "hard", "sep", "cls", "options"]

    def __init__(self, tokenizer: PretrainedTokenizer, max_length: int, prompt: str = None):
        prompt = (
            (
                "{'options': 'choices', 'add_omask': True, 'position': 0, 'token_type': 1}"
                "{'sep': None, 'token_type': 0, 'position': 0}{'text': 'text_a'}{'sep': None, 'token_type': 1}{'text': 'text_b'}"
            )
            if prompt is None
            else prompt
        )
        super(UTCTemplate, self).__init__(prompt, tokenizer, max_length)
        self.max_position_id = self.tokenizer.model_max_length - 1
        self.max_length = max_length
        if not self._has_options():
            raise ValueError(
                "Expected `options` and `add_omask` are in defined prompt, but got {}".format(self.prompt)
            )

    def _has_options(self):
        for part in self.prompt:
            if "options" in part and "add_omask" in part:
                return True
        return False

    def build_inputs_with_prompt(
        self, example: Dict[str, Any], prompt: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        inputs = super(UTCTemplate, self).build_inputs_with_prompt(example, prompt)
        for index, part in enumerate(inputs):
            if "cls" in part:
                inputs[index] = self.tokenizer.cls_token
        return inputs

    def encode(self, example: Dict[str, Any], use_mask: bool = False):
        input_dict = super(UTCTemplate, self).encode(example)

        # Set OMASK and MASK positions and labels for options.
        omask_token_id = self.tokenizer.convert_tokens_to_ids("[O-MASK]")
        input_dict["omask_positions"] = (
            np.where(np.array(input_dict["input_ids"]) == omask_token_id)[0].squeeze().tolist()
        )

        sep_positions = (
            np.where(np.array(input_dict["input_ids"]) == self.tokenizer.sep_token_id)[0].squeeze().tolist()
        )
        input_dict["cls_positions"] = sep_positions[0]

        # Limit the maximum position ids.
        position_ids = np.array(input_dict["position_ids"])
        position_ids[position_ids > self.max_position_id] = self.max_position_id
        input_dict["position_ids"] = position_ids.tolist()

        return input_dict

    def create_prompt_parameters(self):
        return None

    def process_batch(self, input_dict):
        return input_dict
