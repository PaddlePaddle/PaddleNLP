# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import numpy as np


def convert_example(example, tokenizer, is_test=False, language='en'):
    """
    Builds model inputs from a sequence for sequence classification tasks. 
    It use `jieba.cut` to tokenize text.

    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj: paddlenlp.data.JiebaTokenizer): It use jieba to cut the chinese string.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        valid_length(obj:`int`): The input sequence valid length.
        label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
    """
    if is_test:
        input_ids = tokenizer.encode(example["context"])
        valid_length = np.array(len(input_ids), dtype='int64')
        input_ids = np.array(input_ids, dtype='int64')
        return input_ids, valid_length
    else:
        if language == 'en':
            input_ids = tokenizer.encode(example["sentence"])
            label = np.array(example['labels'], dtype="int64")
        else:
            input_ids = tokenizer.encode(example["text"])
            label = np.array(example['label'], dtype="int64")
        valid_length = np.array(len(input_ids), dtype='int64')
        input_ids = np.array(input_ids, dtype='int64')
        return input_ids, valid_length, label


def preprocess_prediction_data(data, tokenizer):
    """
    It process the prediction data as the format used as training.

    Args:
        data (obj:`List[str]`): The prediction data whose each element is  a tokenized text.
        tokenizer(obj: paddlenlp.data.JiebaTokenizer): It use jieba to cut the chinese string.

    Returns:
        examples (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `seq_len`(sequence length).

    """
    examples = []
    for text in data:
        # ids = tokenizer.encode(text)                        # JiebaTokenizer
        ids = tokenizer.encode(text)[0].tolist()[
            1:-1]  # ErnieTokenizer        list[ids]
        examples.append([ids, len(ids)])

    return examples


def get_idx_from_word(word, word_to_idx, unk_word):
    if word in word_to_idx:
        return word_to_idx[word]
    return word_to_idx[unk_word]


class CharTokenizer:

    def __init__(self, vocab, language, vocab_path):
        self.tokenizer = list
        self.vocab = vocab
        self.language = language
        self.vocab_path = vocab_path
        self.unk_token = []

    def encode(self, sentence):
        if self.language == 'ch':
            words = tokenizer_punc(sentence, self.vocab_path)
        else:
            words = sentence.strip().split()
        return [
            get_idx_from_word(word, self.vocab.token_to_idx,
                              self.vocab.unk_token) for word in words
        ]

    def tokenize(self, sentence, wo_unk=True):
        if self.language == 'ch':
            return tokenizer_punc(sentence, self.vocab_path)
        else:
            return sentence.strip().split()

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens)

    def convert_tokens_to_ids(self, tokens):
        return [
            get_idx_from_word(word, self.vocab.token_to_idx,
                              self.vocab.unk_token) for word in tokens
        ]


def tokenizer_lac(string, lac):
    temp = ''
    res = []
    for c in string:
        if '\u4e00' <= c <= '\u9fff':
            if temp != '':
                res.extend(lac.run(temp))
                temp = ''
            res.append(c)
        else:
            temp += c
    if temp != '':
        res.extend(lac.run(temp))
    return res


def tokenizer_punc(string, vocab_path):
    res = []
    sub_string_list = string.strip().split('[MASK]')
    for idx, sub_string in enumerate(sub_string_list):
        temp = ''
        for c in sub_string:
            if '\u4e00' <= c <= '\u9fff':
                if temp != '':
                    temp_seg = punc_split(temp, vocab_path)
                    res.extend(temp_seg)
                    temp = ''
                res.append(c)
            else:
                temp += c
        if temp != '':
            temp_seg = punc_split(temp, vocab_path)
            res.extend(temp_seg)
        if idx < len(sub_string_list) - 1:
            res.append('[MASK]')
    return res


def punc_split(string, vocab_path):
    punc_set = set()
    with open(vocab_path, 'r') as f:
        for token in f:
            punc_set.add(token.strip())
        punc_set.add(' ')
        for ascii_num in range(65296, 65306):
            punc_set.add(chr(ascii_num))
        for ascii_num in range(48, 58):
            punc_set.add(chr(ascii_num))

    res = []
    temp = ''
    for c in string:
        if c in punc_set:
            if temp != '':
                res.append(temp)
                temp = ''
            res.append(c)
        else:
            temp += c
    if temp != '':
        res.append(temp)
    return res
