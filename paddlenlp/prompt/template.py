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
import os
import re
import json

import paddle
import paddle.nn as nn

from .prompt_utils import InputExample, InputFeatures
from .prompt_tokenizer import MLMPromptTokenizer
from ..utils.log import logger

__all__ = ["Template", "ManualTemplate", "SoftTemplate", "AutoTemplate"]

TEMPLATE_FILE = "template.json"


def parse_template(inputs: str, part_start="{", part_end="}"):
    """ Parse items from the input template text. """
    parsed = []
    i_start = 0
    while i_start < len(inputs):
        space = ' ' if (i_start > 0 and inputs[i_start - 1] == ' ') else ''
        p = {"add_prefix_space": space}
        while i_start < len(inputs) and inputs[i_start] == ' ':
            p["add_prefix_space"] = ' '
            i_start += 1
        if i_start == len(inputs): break

        if inputs[i_start] == part_start:
            i_end = i_start + 1
            count_part = 1
            while i_end < len(inputs):
                if inputs[i_end] == part_end:
                    count_part -= 1
                    if count_part == 0: break
                elif inputs[i_end] == part_start:
                    count_part += 1
                i_end += 1
            if i_end == len(inputs):
                raise ValueError(
                    '{} at position {} has no corresponding {}'.format(
                        part_start, i_start, part_end))
            try:
                part = eval('{%s}' % inputs[i_start + 1:i_end])
                if isinstance(part, set):
                    part = {k: None for k in part}
                p.update(part)
            except:
                import traceback
                logger.error(traceback.format_exc())
                logger.error(
                    'syntax error in {}'.format(f"{inputs[i_start + 1:i_end]}"))
                exit()
            i_start = i_end + 1
        else:
            i_end = i_start + 1
            while i_end < len(inputs):
                if inputs[i_end] == part_start:
                    break
                i_end += 1
            p['hard'] = inputs[i_start:i_end].rstrip(' ')
            i_start = i_end
        parsed.append(p)

    return parsed


class Template(nn.Layer):
    """
    Base template class used to preprocess the inputs of model.

    Args:
        tokenizer (paddlenlp.transformers.PretrainedTokenizer):
            The tokenizer of pretrained models.

    """
    registered_input_names = ['mask_ids', 'shortenable_ids']
    registered_text_keys = ['text_a', 'text_b']

    def __init__(self, tokenizer, max_seq_length):
        super().__init__()
        self.tokenizer = tokenizer
        self.wrapped_tokenizer = MLMPromptTokenizer(tokenizer, max_seq_length)

    @property
    def template(self):
        if not hasattr(self, '_template'):
            raise RuntimeError(
                'Property template has not been set before used.')
        return self._template

    @template.setter
    def template(self, template):
        if template is None:
            return
        self._template = template
        self._process_template()

    @abstractmethod
    def _process_template(self):
        """ A hook to process template text when it is set. """
        raise NotImplementedError

    def parse_inputs(self, inputs):
        return parse_template(inputs)

    def get_default_mask_ids(self):
        """ List to denote whether an item in template is a mask token. """
        return [1 if 'mask' in p else 0 for p in self.template]

    def get_default_shortenable_ids(self):
        """ List to denote whther an item in template can be truncated. """
        idx = []
        for p in self.template:
            if 'shortenable' in p:
                idx.append(1 if p['shortenable'] else 0)
            else:
                idx.append(1 if 'text' in p else 0)
        return idx

    def incorporate_template_text(self, example, template=None):
        """ Replace each item in template with real text. """
        inputs = template.copy(
        ) if self.template is None else self.template.copy()

        for i, p in enumerate(inputs):
            if 'text' in p:
                inputs[i] = p['add_prefix_space'] + getattr(example, p['text'])
            elif 'mask' in p:
                inputs[i] = self.tokenizer.mask_token
            elif 'hard' in p:
                inputs[i] = p['add_prefix_space'] + p['hard']
            elif 'sep' in p:
                inputs[i] = self.tokenizer.sep_token
            else:
                raise ValueError('Can not parse {}'.format(p))

        return inputs

    def wrap_one_example(self, example):
        """ Process InputExample according to the predefined template. """
        if self.template is None:
            raise ValueError('The template has not been initialized.')
        if isinstance(example, InputExample):
            text = self.incorporate_template_text(example)

            non_empty_keys = example.keys()
            for key in self.registered_text_keys:
                if key in non_empty_keys:
                    non_empty_keys.remove(key)

            keys, values = ['text'], [text]
            for name in self.registered_input_names:
                keys.append(name)
                v = None
                if hasattr(self, name) and getattr(self, name) is not None:
                    v = getattr(self, name)
                elif hasattr(self, 'get_default_' + name):
                    v = getattr(self, 'get_default_' + name)()
                    setattr(self, name, v)
                else:
                    raise ValueError("""
                        Template's part attribute '{}' is registered but not 
                        initialized. Try using template.{} = [...] to 
                        initialize or create a get_default_{}(self)
                        method in your template.""".format(name, name, name))
                values.append(v)

            wrapped_parts_to_tokenize = []
            for value in list(zip(*values)):
                wrapped_parts_to_tokenize.append(dict(zip(keys, value)))

            wrapped_parts_not_to_tokenize = {
                key: getattr(example, key)
                for key in non_empty_keys
            }
            wrapped_parts_to_tokenize = self.wrapped_tokenizer(
                wrapped_parts_to_tokenize)

            return InputFeatures(**wrapped_parts_to_tokenize,
                                 **wrapped_parts_not_to_tokenize)
        else:
            raise TypeError('InputExample')

    def process_batch(self, batch):
        return batch

    def save_to(self, data_dir):
        with open(os.path.join(data_dir, TEMPLATE_FILE), "w") as f:
            json.dump(self.template, f)


class ManualTemplate(Template):
    """
    ManualTemplate for hard prompt methods, such as PET, EFL.

    Args:
        tokenizer (paddlenlp.transformers.PretrainedTokenizer):
            The same as `Template`.
        template (str | list):
            It describes how to combine text and prompts. For example,
            `str`: "{'text':'text_a'} It is {'mask'}." or a corresponding
            list of dictionary/set parsed by `parse_template` method.
    """

    def __init__(self, tokenizer, max_seq_length, template=None):
        super().__init__(tokenizer=tokenizer, max_seq_length=max_seq_length)
        self.template = template

    def _process_template(self):
        if isinstance(self._template, str):
            self._template = self.parse_inputs(self._template)


class SoftTemplate(Template):
    """
    SoftTemplate on the input layer for soft prompt methods, such as p-tuning.

    Args:
        tokenizer (paddlenlp.transformers.PretrainedTokenizer):
            The same as `Template`.
        template (str | list):
            It describes how to combine text with both manual and soft prompts.
        prompt_encoder (str):
            The encoder to project soft embeddings. Support `lstm` and 'mlp'.
            Use soft embeddings directly when prompt_encoder is `None`. 
    """
    registered_input_names = ['soft_token_ids', 'mask_ids', 'shortenable_ids']

    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 model=None,
                 template=None,
                 prompt_encoder=None,
                 encoder_hidden_size=None):
        super().__init__(tokenizer=tokenizer, max_seq_length=max_seq_length)
        if model is None:
            self.token_embeddings = None
            logger.warning(
                "SoftTemplate: The pretrained model is not given. It would "
                "lead to error unless it is initialized for deployment.")
        else:
            if type(model).__name__.endswith('Model'):
                self.token_embeddings = model.embeddings.word_embeddings
            else:
                for module in model.children():
                    if type(module).__name__.endswith('Model'):
                        self.token_embeddings = module.embeddings.word_embeddings
                        break
            self.token_embeddings.weight.stop_gradient = True
            self.embedding_size = self.token_embeddings.weight.shape[-1]
        self.encoder_hidden_size = encoder_hidden_size
        if self.encoder_hidden_size is not None and prompt_encoder is None:
            logger.warning("`prompt_encoder` is not set yet. Use MLP for "
                           "soft embeddings' projection by default.")
            prompt_encoder = "mlp"
        self.prompt_encoder = prompt_encoder
        self.template = template

    def _process_template(self):
        if isinstance(self._template, str):
            self._template = self.parse_inputs(self._template)
        self.parse_soft_tokens()
        self.generate_parameters()

    @property
    def prompt_encoder(self):
        return self._prompt_encoder

    @prompt_encoder.setter
    def prompt_encoder(self, prompt_encoder):

        if prompt_encoder is None:
            return None

        if getattr(self, "_prompt_encoder", None) is not None:
            logger.warning(
                f"Encoder has already set as {self._prompt_encoder}, change " +
                "`prompt_encoder` will reset parameters.")

        self._prompt_encoder = prompt_encoder

        if self.encoder_hidden_size is None:
            hidden_size = self.embedding_size
        else:
            hidden_size = self.encoder_hidden_size
        if prompt_encoder == 'lstm':
            self.lstm_head = nn.LSTM(input_size=self.embedding_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     direction='bidirect',
                                     time_major=False)
            self.mlp_head = nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, self.embedding_size))
        elif prompt_encoder == 'mlp':
            self.mlp_head = nn.Sequential(
                nn.Linear(self.embedding_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, self.embedding_size))
            if hasattr(self, "lstm_head"):
                delattr(self, "lstm_head")
        else:
            raise ValueError(
                "Unsupported soft token encoder: {}".format(prompt_encoder))

    def incorporate_template_text(self, example, template=None):
        """ Replace each item in template with real text. """
        inputs = template.copy(
        ) if self.template is None else self.template.copy()

        for i, p in enumerate(inputs):
            if 'text' in p:
                inputs[i] = p['add_prefix_space'] + getattr(example, p['text'])
            elif 'mask' in p:
                inputs[i] = self.tokenizer.mask_token
            elif 'hard' in p:
                inputs[i] = p['add_prefix_space'] + p['hard']
            elif 'soft' in p:
                inputs[i] = p['add_prefix_space'] + p['soft']
            elif 'sep' in p:
                inputs[i] = self.tokenizer.sep_token
            else:
                raise ValueError('can not parse {}'.format(p))

        return inputs

    def parse_soft_tokens(self):
        inputs = []
        soft_token_ids = []
        num_soft_token = 0
        soft2word_init = {}
        soft_id_reindex = {}

        for part in self._template:
            if 'soft' not in part and 'soft_id' not in part:
                soft_token_ids.append(0)
                inputs.append(part)
                continue

            if 'soft' in part and part['soft'] is not None:
                if 'duplicate' in part:
                    logger.warning(
                        'Ignore ``duplicate``. It is '
                        'incompatible with ``soft`` with text values.')

                # Get word tokens and ids for soft token initialization.
                init_token_ids = self.tokenizer(
                    part['add_prefix_space'] + part['soft'],
                    add_special_tokens=False,
                    return_token_type_ids=False)['input_ids']
                init_tokens = self.tokenizer.convert_ids_to_tokens(
                    init_token_ids)
                assert len(init_tokens) == len(init_token_ids)

                # Create soft ids and corresponding ``soft`` part in template.
                next_num_soft = num_soft_token + 1
                num_soft_token += len(init_tokens)
                id_list = list(range(next_num_soft, num_soft_token + 1))

                soft_token_ids.extend(id_list)
                inputs.extend([{
                    'add_prefix_space': part['add_prefix_space'],
                    'soft': token
                } for token in init_tokens])
                for soft_id, word_id in zip(id_list, init_token_ids):
                    soft2word_init[soft_id] = word_id

                # Check the ids of ``soft`` and ``soft_id``.
                if 'soft_id' in part:
                    if part['soft_id'] in soft_id_reindex:
                        assert id_list == soft_id_reindex[part['soft_id']]
                    else:
                        soft_id_reindex[part['soft_id']] = id_list
                continue

            if 'soft_id' in part and part['soft_id'] in soft_id_reindex:
                if 'duplicate' in part:
                    logger.warnings('Ignore ``duplicate``. Initialize '
                                    '``soft`` by ``soft_id`` directly.')
                id_list = soft_id_reindex[part['soft_id']]

            elif 'duplicate' in part:
                assert isinstance(part['duplicate'], int)
                if 'same' in part:
                    num_soft_token += 1
                    id_list = [num_soft_token for _ in range(part['duplicate'])]
                else:
                    next_num_soft = num_soft_token + 1
                    num_soft_token += part['duplicate']
                    id_list = list(range(next_num_soft, num_soft_token + 1))
            else:
                num_soft_token += 1
                id_list = [num_soft_token]

            if 'soft_id' in part:
                soft_id_reindex[part['soft_id']] = id_list

            soft_token_ids.extend(id_list)
            inputs.extend([{
                'add_prefix_space': part['add_prefix_space'],
                'soft': self.tokenizer.cls_token
            } for _ in range(len(id_list))])

        self._template = inputs
        self.soft_token_ids = soft_token_ids
        self.num_soft_token = num_soft_token
        self.soft2word_init = soft2word_init

        if self.num_soft_token == 0:
            logger.warning('No soft tokens in template. '\
                'Use ManualTemplate for better performance.')

    def generate_parameters(self):
        """
        Generate parameters for soft tokens.
        """
        if self.num_soft_token == 0 or self.token_embeddings is None:
            return None
        self.soft_embeddings = nn.Embedding(self.num_soft_token + 1,
                                            self.embedding_size)

        weight = self.soft_embeddings.weight.clone().detach()
        for soft_id, word_id in self.soft2word_init.items():
            weight[soft_id] = self.token_embeddings(paddle.to_tensor(word_id))
        self.soft_embeddings.weight.set_value(weight)

    def process_batch(self, batch):
        word_embeds = self.token_embeddings(batch["input_ids"])
        batch["input_ids"] = None
        if not hasattr(self,
                       "soft_embeddings") or batch["soft_token_ids"] is None:
            batch["inputs_embeds"] = word_embeds
        else:
            soft_embeds = self.soft_embeddings(batch["soft_token_ids"])
            if hasattr(self, "lstm_head"):
                soft_embeds = self.lstm_head(soft_embeds)[0]
            if hasattr(self, "mlp_head"):
                soft_embeds = self.mlp_head(soft_embeds)

            inputs_embeds = paddle.where(
                (batch["soft_token_ids"] > 0).unsqueeze(-1), soft_embeds,
                word_embeds)
            batch["inputs_embeds"] = inputs_embeds
        return batch


class AutoTemplate(object):
    """
    AutoTemplate can help you automatically create the relevant Template
    given the provided prompt.
    """
    registered_text_keys = ['text_a', 'text_b']

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            '{} is designed to be instantiated using {}.create_from('\
                'template, tokenizer, text_list, ...)'.format(
                    self.__class__.__name__, self.__class__.__name__))

    @classmethod
    def parse_inputs(cls, inputs):
        return parse_template(inputs)

    @classmethod
    def create_from(cls,
                    template,
                    tokenizer,
                    max_seq_length,
                    model=None,
                    prompt_encoder=None,
                    encoder_hidden_size=None):
        if template is None:
            template = "{'soft'}"
        if isinstance(template, str):
            template = cls.parse_inputs(template)
        template_keys = cls._extract_template_keys(template)
        if 'text' not in template_keys:
            soft_template = []
            for item in template:
                if 'hard' in item:
                    soft_template.append({
                        'add_prefix_space': '',
                        'soft': item['hard']
                    })
                else:
                    soft_template.append(item)
            text_item = [{
                'add_prefix_space': ' ',
                'text': cls.registered_text_keys[0]
            }]
            template = text_item + soft_template
        template_keys = cls._extract_template_keys(template)

        if 'mask' not in template_keys:
            template.append({'add_prefix_space': ' ', 'mask': None})

        if 'soft' in template_keys:
            return SoftTemplate(tokenizer=tokenizer,
                                template=template,
                                max_seq_length=max_seq_length,
                                model=model,
                                prompt_encoder=prompt_encoder,
                                encoder_hidden_size=encoder_hidden_size)
        else:
            return ManualTemplate(tokenizer=tokenizer,
                                  max_seq_length=max_seq_length,
                                  template=template)

    @classmethod
    def load_from(cls,
                  data_dir,
                  tokenizer,
                  max_seq_length,
                  model=None,
                  prompt_encoder=None,
                  encoder_hidden_size=None):
        with open(os.path.join(data_dir, TEMPLATE_FILE), "r") as f:
            template = json.load(f)
        return cls.create_from(template, tokenizer, max_seq_length, model,
                               prompt_encoder, encoder_hidden_size)

    @classmethod
    def _extract_template_keys(cls, inputs: list):
        template_keys = set()
        for item_dict in inputs:
            for key, value in item_dict.items():
                template_keys.add(key)
                if key == 'text':
                    assert value in cls.registered_text_keys, 'No ``{}`` attribute '\
                        'in InputExample.'.format(value)
        return template_keys
