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

import itertools
import warnings
from functools import partial
from collections import defaultdict

import numpy as np
from paddlenlp.utils.log import logger


class TokenizerWrapper:
    """
    Process examples encoded by template, such as truncating and padding.

    Args:
        max_seq_length (int):
            The maximum length of input data (prompt and text).
        tokenizer (paddlenlp.transformers.PreTrainedTokenizer):
            The tokenizer of pretrained model.
        truncate_method (str):
            How to truncate input data. 
            Choices: ``tail``, ``head``, ``manual``.
        create_token_type_ids (bool):
            Whether to create token_type_ids for inputs.
        seq_length_list (list, optional):
            The list of maximum length for every part in input data.  
    """

    def __init__(self,
                 max_seq_length,
                 tokenizer,
                 truncate_method='tail',
                 create_token_type_ids=False,
                 **kwargs):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        if truncate_method == 'manual':
            assert hasattr(kwargs, 'seq_length_list'), 'seq_length_list '\
                'should be defined for manual truncation.'
            self.seq_length_list = kwargs['seq_length_list']
            self.truncate_fn = partial(self.truncate_from_end, etype='tail')
        elif truncate_method == 'tail' or truncate_method == 'head':
            self.truncate_fn = partial(self.truncate_from_end,
                                       etype=truncate_method)
        else:
            raise NotImplementedError

        self.create_token_type_ids = create_token_type_ids

        self.num_truncated_sentences = 0
        self.total_passed_sentences = 0

    @property
    def special_tokens_maps(self):
        if not hasattr(self, "_special_tokens_map"):
            self._special_tokens_map = {
                '<cls>': getattr(self.tokenizer, 'cls_token', ''),
                '<sep>': getattr(self.tokenizer, 'sep_token', ''),
                '<pad>': getattr(self.tokenizer, 'pad_token', ''),
                '<mask>': getattr(self.tokenizer, 'mask_token', ''),
                '<unk>': getattr(self.tokenizer, 'unk_token', '')
            }
        return self._special_tokens_map

    @property
    def truncate_rate(self):
        if self.total_passed_sentences == 0:
            return None
        else:
            return self.num_truncated_sentences / self.total_passed_sentences

    @staticmethod
    def truncate_by_manual(input_dict, max_len_list=[]):
        """
        Truncate input data by manually defined maximum sequence length.

        Args:
            input_dict (dict):
                The dictionary of an input example.
            max_len_list (list):
                The maximum length of every part in example.
                ``-1`` denotes that there is no limit on length.
        """
        truncated_dict = defaultdict(list)
        shortenable_ids = input_dict['shortenable_ids']
        truncated_dict['shortenable_ids'] = shortenable_ids
        for attr_name, attr_values in input_dict.items():
            text_idx = 0
            for i, value in enumerate(attr_values):
                if shortenable_ids[i][0] == 0:
                    continue
                if text_idx >= len(max_len_list):
                    break
                if len(value) > 0:
                    max_len = max_len_list[text_idx]
                    if max_len < 0:
                        attr_values[i] = value
                    else:
                        attr_values[i] = value[:max_len]
                text_idx += 1
            truncated_dict[attr_name] = attr_values
        return truncated_dict

    @staticmethod
    def truncate_from_end(input_dict, num_tokens_to_truncate=0, etype='tail'):
        assert etype in ['head', 'tail']
        step = 1 if etype == 'head' else -1
        idx_offset = 0 if etype == 'head' else 1
        truncated_dict = defaultdict(list)
        shortenable_ids = input_dict['shortenable_ids']
        for attr_name in input_dict:
            attr_values = input_dict[attr_name]
            count = num_tokens_to_truncate
            for i, value in enumerate(attr_values[::step]):
                index = int(step * (idx_offset + i))
                if len(value) == 0 or shortenable_ids[index][0] == 0:
                    continue
                if count < len(value):
                    attr_values[index] = value[:-count]
                else:
                    attr_values[index] = []
                count -= len(value)
                if count <= 0:
                    break
            truncated_dict[attr_name] = attr_values

        return truncated_dict

    @staticmethod
    def concate_parts(input_dict):
        for key in input_dict:
            input_dict[key] = list(itertools.chain(*input_dict[key]))
        return input_dict

    @staticmethod
    def padding(input_dict,
                max_len,
                pad_id_for_inputs=0,
                pad_id_for_others: int = 0) -> None:
        for key, value in input_dict.items():
            if (len(input_dict[key]) > max_len):
                raise ValueError(
                    f'''Truncated seq length of '{key}' still greater than 
                    max length {max_len}. One possible reason is that 
                    no enough shortenable parts in template. Try adding
                    {{"shortenable": "True"}} property.
                ''')
            if 'input' in key:
                input_dict[key].extend([pad_id_for_inputs] *
                                       (max_len - len(value)))
            else:
                input_dict[key].extend([pad_id_for_others] *
                                       (max_len - len(value)))
        return input_dict

    def truncate(self, inputs):
        if hasattr(self, 'seq_length_list'):
            inputs = self.truncate_by_manual(inputs, self.seq_length_list)
        total_tokens = sum([len(part) for part in inputs['input_ids']])
        num_specials = self.num_special_tokens_to_add
        num_tokens_to_truncate = total_tokens - self.max_seq_length + num_specials
        self.total_passed_sentences += 1
        if num_tokens_to_truncate > 0:
            self.num_truncated_sentences += 1
            inputs = self.truncate_fn(
                input_dict=inputs,
                num_tokens_to_truncate=num_tokens_to_truncate)
        return inputs

    def add_special_tokens(self, encode_inputs):
        for key in encode_inputs:
            if key == "input_ids":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    encode_inputs[
                        key] = self.tokenizer.build_inputs_with_special_tokens(
                            encode_inputs[key])
            else:
                special_tokens_mask = np.array(
                    self.tokenizer.get_special_tokens_mask(encode_inputs[key]))
                with_special_tokens = np.array(
                    self.tokenizer.build_inputs_with_special_tokens(
                        encode_inputs[key]))
                with_special_tokens[special_tokens_mask == 1] = 0
                encode_inputs[key] = with_special_tokens.tolist()
        return encode_inputs


class MLMTokenizerWrapper(TokenizerWrapper):
    input_keys = ['input_ids', 'attention_mask', 'token_type_ids']

    @property
    def mask_token(self):
        return self.tokenizer.mask_token

    @property
    def mask_token_id(self):
        return self.tokenizer.mask_token_id

    @property
    def soft_token(self):
        return self.tokenizer.unk_token

    @property
    def soft_token_id(self):
        return self.tokenizer.unk_token_id

    @property
    def num_special_tokens_to_add(self):
        if not hasattr(self, '_num_specials'):
            self._num_specials = self.tokenizer.num_special_tokens_to_add()
        return self._num_specials

    def get_token_type_ids(self, encoded_inputs):
        token_type_ids = [0] * len(encode_inputs['input_ids'])
        sep_token = getattr(self.tokenizer, 'sep_token', -1)
        if sep_token >= 0:
            sep_index = np.where(
                [x == sep_token for x in encode_inputs['input_ids']])[0]
            for i, x in enumerate(sep_index[1:]):
                pre_x = sep_index[i - 1]
                sep_index[pre_x + 1:x + 1] = [i + 1] * (x - pre_x)
        return token_type_ids

    def tokenize_one_example(self, wrapped_example):
        to_tokenize, not_to_tokenize = wrapped_example

        encode_inputs = defaultdict(list)
        for part in to_tokenize:
            if part['mask_ids'] == 1:
                text = [self.mask_token_id]

            if part['text'] in self.special_tokens_maps.keys():
                to_replace = self.special_tokens_maps[part['text']]
                if to_replace is not None:
                    part['text'] = to_replace
                else:
                    raise KeyError(
                        "This tokenizer doesn't specify {} token.".format(
                            piece['prompt']))

            if 'soft_token_ids' in part and part['soft_token_ids'] == 1:
                text = [self.soft_token_id]
            else:
                text = self.tokenizer.encode(
                    part['text'],
                    add_special_tokens=False,
                    return_token_type_ids=False)['input_ids']

            text_len = len(text)
            encode_inputs['input_ids'].append(text)
            for key in part:
                if key not in ['text']:
                    encode_inputs[key].append([part[key]] * text_len)
        encode_inputs = self.truncate(inputs=encode_inputs)
        encode_inputs.pop('shortenable_ids')
        encode_inputs = self.concate_parts(encode_inputs)
        encode_inputs = self.add_special_tokens(encode_inputs)
        encode_inputs['attention_mask'] = [1] * len(encode_inputs['input_ids'])
        if self.create_token_type_ids:
            encode_inputs['token_type_ids'] = get_token_type_ids(encode_inputs)
        encode_inputs = self.padding(
            encode_inputs,
            max_len=self.max_seq_length,
            pad_id_for_inputs=self.tokenizer.pad_token_id)

        return {**encode_inputs}


tokenizer_mapping = {
    'roberta': MLMTokenizerWrapper,
}
