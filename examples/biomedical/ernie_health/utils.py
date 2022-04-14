# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

from paddlenlp.transformers import convert_to_unicode, normalize_chars, tokenize_special_chars

# Special characters found in baidu medical corpus.
SPECIAL_MEDICAL_CHARACTER = [
    248, 713, 711, 305, 720, 712, 4523, 4449, 4469, 716, 4540, 4457, 4462, 4527,
    4453, 230, 4363, 8322, 4520, 4352, 4364, 240, 4450, 4361, 322, 4467, 714,
    223, 4370, 4355, 8323, 4458, 4354, 1649, 12640, 12832, 7506, 7499, 4454,
    4366, 1637, 190, 43915, 12557, 12554, 8321, 7500, 7447, 4455, 4451, 4357,
    2134, 715, 254, 42603, 41594, 12841, 12840, 12615, 12567, 8319, 8310, 7508,
    7502, 6980, 6508, 5604, 5465, 4536, 4535, 4468, 4461, 4359, 4314, 3665,
    3648, 3376, 3104, 3090, 3056, 2752, 1726, 689, 179, 43903, 43689, 12839,
    12644, 12579, 12556, 12327, 12294, 12293, 8320, 7491, 5711, 5608, 5596,
    5154, 5141, 4539, 4538, 4326, 3623, 3092, 2920, 1782, 1641, 1575, 748, 436,
    295, 170, 43931, 43739, 43712, 43682, 43681, 43677, 42601, 41983, 41512,
    41485, 12636, 12613, 12609, 12570, 8413, 8324, 8311, 7576, 7557, 7511, 7509,
    7503, 7461, 6965, 6929, 6846, 5603, 5589, 5586, 5463, 5281, 4756, 4545,
    4525, 4460, 4358, 4317, 3776, 3634, 3588, 3515, 3495, 3286, 3267, 3250,
    3248, 3178, 2979, 2791, 2751, 2663, 1709, 1602, 447, 402, 189, 188, 185,
    120116, 78239, 64258, 64257, 43967, 43962, 43960, 43936, 43929, 43917,
    43888, 43688, 41857, 41482, 41460, 41001, 12835, 12833, 12581, 12576, 12574,
    12572, 12569, 12566, 12563, 12560, 12559, 12552, 12550, 12328, 12325, 12322,
    8543, 8531, 8466, 8304, 7567, 7518, 7512, 7505, 7497, 7457, 7185, 6975,
    6948, 6931, 6675, 5602, 5478, 5464, 5461, 5457, 5355, 5337, 5334, 5314,
    5195, 5166, 5158, 4640, 4546, 4537, 4526, 4463, 4459, 4367, 4365, 4360,
    4353, 4304, 4139, 4114, 3940, 3928, 3779, 3778, 3632, 3589, 3431, 3367,
    3309, 3266, 3176, 2995, 2919, 2867, 2866, 2846, 2792, 2749, 2699, 2537,
    2441, 2415, 2339, 2327, 2125, 2088, 2053, 1882, 1810, 1788, 1736, 1723,
    1657, 1610, 1591, 1587, 1585, 1583, 1576, 1521, 1512, 1508, 1507, 1391,
    1387, 1295, 717, 697, 696, 688, 339, 307, 65228, 43686, 43683, 41538, 12838,
    12837, 12836, 12834, 12671, 12622, 12621, 12610, 12585, 12584, 12583, 12582,
    12580, 12575, 12562, 12561, 12555, 12553, 12551, 12323, 8578, 8313, 7446,
    7437, 7431, 5528, 5527, 5147, 3904, 3667, 3664, 3591, 3585, 3526, 3237,
    3233, 3232, 3109, 2951, 2840, 2835, 2797, 2669, 2553, 2407, 1697, 1608,
    1607, 1390, 1285, 710, 426
]

# 1405 emoji characters parsed from emoji v14.0 (https://unicode.org/Public/emoji/14.0/emoji-test.txt).
# The parse function refers to https://github.com/UWPX/Emoji-List-Parser/blob/master/emoji_parser.py.
# Any character belongs to Basic Latin is removed (from the 'keycap' subgroup).
with open('emoji_codepoints.txt', 'r') as fp:
    EMOJI_CODEPOINTS = [int(x.strip()) for x in fp.readlines()]


class PreTokenizer(object):
    """
    Runs basic tokenization (punctuation splitting, lower casing, etc.).
    Args:
        do_lower_case (bool): Whether the text strips accents and convert to
            lower case. If you use the BERT Pretrained model, lower is set to
            Flase when using the cased model, otherwise it is set to True.
            Default: True.
    """

    def __init__(self,
                 do_normalize=True,
                 do_split_special_chars=True,
                 do_split_number=True):
        """Constructs a BasicTokenizer."""

        self.do_normalize = True
        self.do_split_special_chars = True
        self.do_split_number = True

    def pre_tokenize(self, text):
        """
        Tokenizes a piece of text using basic tokenizer.
        Args:
            text (str): A piece of text.
        Returns:
            list(str): A list of tokens.
        """
        text = convert_to_unicode(text)
        if self.do_normalize:
            text = normalize_chars(text)
        if self.do_split_special_chars:
            text = self._split_special_chars(text)
        if self.do_split_number:
            text = self._split_number(text)

        return text

    def _split_number(self, text):
        """
        Performs invalid character removal and whitespace cleanup on text.
        """
        output = []
        for i, char in enumerate(text):
            if self._is_number_char(char):
                if i > 0 and not self._is_number_char(text[i - 1]):
                    output.append(" ")
                output.append(char)
                if i < len(text) - 1 and not self._is_number_char(text[i + 1]):
                    output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_number_char(self, char):
        """
        Checks whether CP is the codepoint of a arabic number.
        """
        cp = ord(char)
        if 0x0030 <= cp <= 0x0039:
            return True
        return False

    def _split_special_chars(self, text):
        text = tokenize_special_chars(text)
        output = []
        for char in text:
            cp = ord(char)
            if (cp in SPECIAL_MEDICAL_CHARACTER
                    or  # Special characters found in medical corpus
                    cp in EMOJI_CODEPOINTS):  # Emoji
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)
