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

import unicodedata
import numpy as np


def _normalize(text):
    """
    Normalize the text for multiligual and chinese models. Unicode range:
    https://www.ling.upenn.edu/courses/Spring_2003/ling538/UnicodeRanges.html
    """
    output = []
    for char in text:
        cp = ord(char)
        if (0xFF00 <= cp <= 0xFFEF or  # Halfwidth and Fullwidth Forms
                0xFE50 <= cp <= 0xFE6B or  # Small Form Variants
                0x3358 <= cp <= 0x33FF or  # CJK Compatibility
                0x249C <= cp <= 0x24E9):  # Enclosed Alphanumerics: Ⓛ ⒰
            for c in unicodedata.normalize('NFKC', char):
                output.append(c)
        elif (0x2460 <= cp <= 0x249B or 0x24EA <= cp <= 0x24FF or
              0x2776 <= cp <= 0x2793 or  # Enclosed Alphanumerics
              0x2160 <= cp <= 0x217F):  # Number Forms
            output.append(' ')
            for c in str(int(unicodedata.numeric(char))):
                output.append(c)
            output.append(' ')
        elif cp == 0xF979:  # https://www.zhihu.com/question/20697984
            output.append('凉')
        else:
            output.append(char)
    return "".join(output)


def _tokenize_special_chars(text):
    output = []
    for char in text:
        cp = ord(char)
        if (0x3040 <= cp <= 0x30FF or  # Japanese
                0x0370 <= cp <= 0x04FF or  # Greek/Coptic & Cyrillic
                0x0250 <= cp <= 0x02AF or  # IPA
                cp in
            [0x00ad, 0x00b2, 0x00ba, 0x3007, 0x00b5, 0x00d8, 0x014b, 0x01b1] or
                unicodedata.category(char).startswith('S')):  # Symbol
            output.append(' ')
            output.append(char)
            output.append(' ')
        else:
            output.append(char)
    return ''.join(output)


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    """
    Builds model inputs from a sequence or a pair of sequences for sequence
    classification tasks by concatenating and adding special tokens. And
    creates a mask from the two sequences for sequence-pair classification
    tasks.

    The convention in Electra/EHealth is:

    - single sequence:
        input_ids:      ``[CLS] X [SEP]``
        token_type_ids: ``  0   0   0``
        position_ids:   ``  0   1   2``

    - a senquence pair:
        input_ids:      ``[CLS] X [SEP] Y [SEP]``
        token_type_ids: ``  0   0   0   1   1``
        position_ids:   ``  0   1   2   3   4``

    Args:
        example (obj:`dict`):
            A dictionary of input data, containing text and label if it has.
        tokenizer (obj:`PretrainedTokenizer`):
            A tokenizer inherits from :class:`paddlenlp.transformers.PretrainedTokenizer`.
            Users can refer to the superclass for more information.
        max_seq_length (obj:`int`):
            The maximum total input sequence length after tokenization.
            Sequences longer will be truncated, and the shorter will be padded.
        is_test (obj:`bool`, default to `False`):
            Whether the example contains label or not.

    Returns:
        input_ids (obj:`list[int]`):
            The list of token ids.
        token_type_ids (obj:`list[int]`):
            List of sequence pair mask.
        position_ids (obj:`list[int]`):
            List of position ids.
        label(obj:`numpy.array`, data type of int64, optional):
            The input label if not is_test.
    """
    text_a = example['text_a']
    text_b = example.get('text_b', None)

    text_a = _tokenize_special_chars(_normalize(text_a))
    if text_b is not None:
        text_b = _tokenize_special_chars(_normalize(text_b))

    encoded_inputs = tokenizer(
        text=text_a,
        text_pair=text_b,
        max_seq_len=max_seq_length,
        return_position_ids=True)
    input_ids = encoded_inputs['input_ids']
    token_type_ids = encoded_inputs['token_type_ids']
    position_ids = encoded_inputs['position_ids']

    if is_test:
        return input_ids, token_type_ids, position_ids
    label = np.array([example['label']], dtype='int64')
    return input_ids, token_type_ids, position_ids, label
