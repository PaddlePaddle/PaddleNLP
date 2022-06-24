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
import time
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.utils.log import logger
from data import InputExample


class Template(nn.Layer):
    """
    Base template class used to preprocess the inputs of model.

    Args:
        tokenizer (paddlenlp.transformers.PretrainedTokenizer):
            The tokenizer of pretrained models.
        text_mapping (dict):
            The dictionary to map text name in template to that in InputExample.
            For example, {'premise': 'text_a', 'hypothesis': 'text_b'}.

    """
    registered_input_names = ['mask_ids', 'shortenable_ids']

    def __init__(self, tokenizer, text_mapping=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_mapping = text_mapping
        self._process_lock = False

        self.part_start = '{'
        self.part_end = '}'

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
        self.process_template()

    @abstractmethod
    def process_template(self):
        """ A hook to process template text when it is set. """
        raise NotImplementedError

    def get_default_mask_ids(self):
        """ List to denote whether an item in template is a mask token. """
        return [1 if 'mask' in p else 0 for p in self.template]

    def get_default_shortenable_ids(self):
        """ List to denote whther an item in template can be truncated. """
        idx = []
        for p in self.template:
            if 'shortenable' in p:
                idx.append(1 if d['shortenable'] else 0)
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
            else:
                raise ValueError('can not parse {}'.format(p))

        return inputs

    def parse_inputs(self, inputs: str):
        """ Parse items from the input template text. """
        parsed = []
        i = 0
        while i < len(inputs):
            p = {
                'add_prefix_space': ' ' if
                (i > 0 and inputs[i - 1] == ' ') else ''
            }
            while i < len(inputs) and inputs[i] == ' ':
                p['add_prefix_space'] = ' '
                i = i + 1
            if i == len(inputs): break

            if inputs[i] == self.part_start:
                j = i + 1
                count_part = 1
                while j < len(inputs):
                    if inputs[j] == self.part_end:
                        count_part -= 1
                        if count_part == 0: break
                    elif inputs[j] == self.part_start:
                        count_part += 1
                    j = j + 1
                if j == len(inputs):
                    raise ValueError(
                        '{} at position {} has no corresponding {}'.format(
                            self.part_start, i, self.part_end))
                try:
                    part = eval('{%s}' % inputs[i + 1:j])
                    if isinstance(part, set):
                        part = {k: None for k in part}
                    p.update(part)
                except:
                    import traceback
                    logger.error(traceback.format_exc())
                    logger.error('syntax error in {}'.format('{%s}' %
                                                             inputs[i + 1:j]))
                    exit()
                i = j + 1
            else:
                j = i + 1
                while j < len(inputs):
                    if inputs[j] == self.part_start:
                        break
                    j = j + 1
                p['hard'] = inputs[i:j].rstrip(' ')
                i = j
            parsed.append(p)

        return parsed

    def wrap_one_example(self, example):
        """ Process InputExample according to the predefined template. """
        if self.template is None:
            raise ValueError('template has not been initialized.')
        if isinstance(example, InputExample):
            text = self.incorporate_template_text(example)

            non_empty_keys = example.keys()
            for key in self.text_mapping:
                if self.text_mapping[key] in non_empty_keys:
                    non_empty_keys.remove(self.text_mapping[key])

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
                        Template's part attribute '{}' is registered but not initialized.
                        Try using template.{} = [...] to initialize or create a get_default_{}(self)
                        method in your template.""".format(name, name, name))
                values.append(v)

            wrapped_parts_to_tokenize = []
            for value in list(zip(*values)):
                wrapped_parts_to_tokenize.append(dict(zip(keys, value)))

            wrapped_parts_not_to_tokenize = {
                key: getattr(example, key)
                for key in non_empty_keys
            }
            return [wrapped_parts_to_tokenize, wrapped_parts_not_to_tokenize]
        else:
            raise TypeError('InputExample')


class ManualTemplate(Template):
    """
    ManualTemplate for hard prompt methods, such as PET, EFL.
    """

    def __init__(self,
                 tokenizer,
                 template=None,
                 text_mapping={
                     'text_a': 'text_a',
                     'text_b': 'text_b'
                 }):
        super().__init__(tokenizer=tokenizer, text_mapping=text_mapping)
        self.template = template

    def process_template(self):
        self._template = self.parse_inputs(self._template)


class SoftTemplate(Template):
    """
    SoftTemplate for soft prompt methods, such as prefix-tuning, p-tuning.
    """

    def __init__(self,
                 tokenizer,
                 model=None,
                 template=None,
                 text_mapping={
                     'text_a': 'text_a',
                     'text_b': 'text_b'
                 },
                 soft_embeddings=None):
        super().__init__(tokenizer=tokenizer, text_mapping=text_mapping)
        self.template = template
        self.tokenizer = tokenizer
        self.token_embeddings = model.embeddings
        self.embedding_size = self.token_embeddings.weight.shape[-1]

        if soft_embeddings is not None:
            self.soft_embeddings = soft_embeddings
            self.num_soft_token = soft_embeddings.shape[0]

    def process_template(self):
        self._template = self.parse_inputs(self._template)
        self.process_soft_tokens()

    def process_soft_tokens(self):
        inputs = []
        soft_ids = []
        num_soft_token = 0
        soft2word_init = {}
        soft_id_reindex = {}

        for part in self.template:
            if 'soft' not in part and 'soft_id' not in part:
                soft_ids.append(0)
                inputs.append(part)
                continue

            if 'soft' in part and part['soft'] is not None:
                if 'duplicate' in part:
                    logger.warnings(
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

                soft_ids.extend(id_list)
                inputs.extend([{'soft': token} for token in init_tokens])
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

            soft_ids.extend(id_list)
            inputs.extend([{'soft': None} for _ in range(len(id_list))])

        self._template = inputs
        self.soft_ids = soft_ids
        self.num_soft_token = num_soft_token
        self.soft2word_init = soft2word_init

        if hasattr(self, 'soft_embeddings'):
            assert self.num_soft_token == self.soft_embeddings.shape[0]
        else:
            self.soft_embeddings = nn.Embedding(self.num_soft_token,
                                                self.embedding_size)

        weight = self.soft_embeddings.weight.clone().detach()
        for soft_id, word_id in self.soft2word_init.items():
            weight[soft_id] = self.token_embeddings(word_id)
        self.soft_embeddings.weight.set_value(weight)
