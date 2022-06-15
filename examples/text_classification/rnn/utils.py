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
from collections import defaultdict
import re

from paddlenlp import Taskflow
import numpy as np

word_segmenter = Taskflow("word_segmentation", mode="fast")


def convert_example(example, tokenizer, is_test=False):
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

    input_ids = tokenizer.encode(example["text"])
    valid_length = np.array(len(input_ids), dtype='int64')
    input_ids = np.array(input_ids, dtype='int64')

    if not is_test:
        label = np.array(example["label"], dtype="int64")
        return input_ids, valid_length, label
    else:
        return input_ids, valid_length


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
        ids = tokenizer.encode(text)
        examples.append([ids, len(ids)])
    return examples


def build_vocab(texts,
                stopwords=[],
                num_words=None,
                min_freq=10,
                unk_token="[UNK]",
                pad_token="[PAD]"):
    """
    According to the texts, it is to build vocabulary.

    Args:
        texts (obj:`List[str]`): The raw corpus data.
        num_words (obj:`int`): the maximum size of vocabulary.
        stopwords (obj:`List[str]`): The list where each element is a word that will be
            filtered from the texts.
        min_freq (obj:`int`): the minimum word frequency of words to be kept.
        unk_token (obj:`str`): Special token for unknow token.
        pad_token (obj:`str`): Special token for padding token.

    Returns:
        word_index (obj:`Dict`): The vocabulary from the corpus data.

    """
    word_counts = defaultdict(int)
    for text in texts:
        if not text:
            continue
        for word in word_segmenter(text):
            if word in stopwords:
                continue
            word_counts[word] += 1

    wcounts = []
    for word, count in word_counts.items():
        if count < min_freq:
            continue
        wcounts.append((word, count))
    wcounts.sort(key=lambda x: x[1], reverse=True)
    # -2 for the pad_token and unk_token which will be added to vocab.
    if num_words is not None and len(wcounts) > (num_words - 2):
        wcounts = wcounts[:(num_words - 2)]
    # add the special pad_token and unk_token to the vocabulary
    sorted_voc = [pad_token, unk_token]
    sorted_voc.extend(wc[0] for wc in wcounts)
    word_index = dict(zip(sorted_voc, list(range(len(sorted_voc)))))
    return word_index
