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

import pickle
import re

from paddlenlp.transformers import BasicTokenizer
from paddlenlp.transformers.tokenizer_utils import (
    _is_control,
    _is_punctuation,
    _is_whitespace,
    is_chinese_char,
    tokenize_special_chars,
)

re_eng = re.compile("[#a-zA-Z0-9]", re.U)
re_sep = re.compile("\[[A-Z]+\]", re.U)
re_sep_eng = re.compile("\<[\/a-z]+\>", re.U)

bt = BasicTokenizer()
normalize_chars = lambda x: "".join(bt.tokenize(x))


# 20902 个中文全字符
def chinese_char():
    return set([chr(x) for x in range(0x4E00, 0x9FA5 + 1)])


# 日文 或 谚文字母
def jk_vocab(c):
    c = ord(c)
    return (c >= 0x3040 and c <= 0x33FF) or (c >= 0x1100 and c <= 0x11FF)  # 谚文字母


# 特殊 TOKEN
def add_special_token():
    return ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]


char_dict = pickle.load(open("char_dict.pickle", "rb"))
chinese_vocab = chinese_char()
final_vocab = set()
other_char = []


def add_vocab(char, f):
    if re_sep_eng.match(char):
        # Delete <pad> tokens in eng.vocab
        return

    # Add eng vocab and specical token
    if re_eng.match(char) or re_sep.match(char):
        if char not in final_vocab:
            final_vocab.add(char)
            f.write(f"{char}\n")
        return

    # Add chinese char
    if len(char) > 1 and char.startswith("##") and chinese_vocab(char[2]):
        if char not in final_vocab:
            final_vocab.add(char)
            f.write(f"{char}\n")
        return

    # Normalize char， 部分字符 nioe
    char = normalize_chars(char)
    for i, k in enumerate(char):
        if _is_whitespace(k) or _is_control(k):
            continue
        if k not in final_vocab:
            if not _is_punctuation(k) and not is_chinese_char(ord(k)) and k == tokenize_special_chars(k):
                other_char.append(k)
            final_vocab.add(k)
            f.write(f"{k}\n")
            if jk_vocab(k):
                # add "##" for japanese and korean char
                add_vocab("##" + k, f)


with open("vocab.txt", "w") as f:
    for x in add_special_token():
        add_vocab(x, f)

    res = sorted(char_dict.items(), key=lambda x: -x[1])

    # Add chinse char by freq
    for x in res:
        k, v = x
        k = normalize_chars(k)
        if k in chinese_vocab:
            add_vocab(k, f)
            chinese_vocab.remove(k)

    # If chinse char not in freq add it
    chinese_vocab = sorted(chinese_vocab)
    while len(chinese_vocab) > 0:
        k = chinese_vocab.pop()
        if k not in final_vocab:
            f.write(f"{k}\n")
            final_vocab.add(k)

    # And english vocab part
    with open("eng.vocab") as ec:
        line = ec.readline()
        while line:
            k, v = line.strip().split()
            if "▁" in k:
                # remove "▁" in eng vocab
                k = k[1:]
            elif re_sep_eng.match(k):
                pass
            else:
                # add "##" for eng vocab
                k = "##" + k

            add_vocab(k, f)
            line = ec.readline()

    # Add additional tokens in corpus
    # such as japanese and korean char and other symbols
    for x in res:
        k, v = x
        if v >= 200:
            add_vocab(k, f)
