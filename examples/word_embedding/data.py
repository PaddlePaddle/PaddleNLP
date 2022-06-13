# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddlenlp.data import JiebaTokenizer, Vocab
import jieba

tokenizer = jieba


def set_tokenizer(vocab):
    global tokenizer
    if vocab is not None:
        tokenizer = JiebaTokenizer(vocab=vocab)


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n").split("\t")[0]
        vocab[token] = index
    return vocab


def convert_tokens_to_ids(tokens, vocab):
    """ Converts a token id (or a sequence of id) in a token string
        (or a sequence of tokens), using the vocabulary.
    """

    ids = []
    unk_id = vocab.get('[UNK]', None)
    for token in tokens:
        wid = vocab.get(token, unk_id)
        if wid:
            ids.append(wid)
    return ids


def convert_example(example, vocab, unk_token_id=1, is_test=False):
    """
    Builds model inputs from a sequence for sequence classification tasks. 
    It use `jieba.cut` to tokenize text.
    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        vocab(obj:`dict`): The vocabulary.
        unk_token_id(obj:`int`, defaults to 1): The unknown token id.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.
    Returns:
        input_ids(obj:`list[int]`): The list of token ids.s
        valid_length(obj:`int`): The input sequence valid length.
        label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
    """

    input_ids = []
    for token in tokenizer.cut(example['text']):
        token_id = vocab.get(token, unk_token_id)
        input_ids.append(token_id)
    valid_length = np.array([len(input_ids)])
    input_ids = np.array(input_ids, dtype="int32")
    if not is_test:
        label = np.array(example["label"], dtype="int64")
        return input_ids, valid_length, label
    else:
        return input_ids, valid_length


def pad_texts_to_max_seq_len(texts, max_seq_len, pad_token_id=0):
    """
    Padded the texts to the max sequence length if the length of text is lower than it.
    Unless it truncates the text.
    Args:
        texts(obj:`list`): Texts which contrains a sequence of word ids.
        max_seq_len(obj:`int`): Max sequence length.
        pad_token_id(obj:`int`, optinal, defaults to 0) : The pad token index.
    """
    for index, text in enumerate(texts):
        seq_len = len(text)
        if seq_len < max_seq_len:
            padded_tokens = [pad_token_id for _ in range(max_seq_len - seq_len)]
            new_text = text + padded_tokens
            texts[index] = new_text
        elif seq_len > max_seq_len:
            new_text = text[:max_seq_len]
            texts[index] = new_text


def preprocess_prediction_data(data, vocab):
    """
    It process the prediction data as the format used as training.
    Args:
        data (obj:`List[str]`): The prediction data whose each element is  a tokenized text.
    Returns:
        examples (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `seq_len`(sequence length).
    """
    examples = []
    for text in data:
        tokens = " ".join(tokenizer.cut(text)).split(' ')
        ids = convert_tokens_to_ids(tokens, vocab)
        examples.append([ids, len(ids)])
    return examples
