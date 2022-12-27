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

import random
import re

# from fengshen.examples.pegasus.pegasus_utils import text_segmentate
import sys
import unicodedata

import numpy as np
import rouge
import six
import torch

sys.path.append("../../../")

rouge = rouge.Rouge()


is_py2 = six.PY2

if not is_py2:
    basestring = str


def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)
        or (cp >= 0x20000 and cp <= 0x2A6DF)
        or (cp >= 0x2A700 and cp <= 0x2B73F)
        or (cp >= 0x2B740 and cp <= 0x2B81F)
        or (cp >= 0x2B820 and cp <= 0x2CEAF)
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)
    ):
        return True

    return False


def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def is_string(s):
    """判断是否是字符串"""
    return isinstance(s, basestring)


def is_stopwords(word, stopwords):
    if word in stopwords:
        return True
    else:
        return False


def text_segmentate(text):
    en_seg_pattern = "((?:\\!|\\?|\\.|\\n)+(?:\\s)+)"
    ch_seg_pattern = "((?:？|！|。|\\n)+)"
    try:
        text = re.sub(en_seg_pattern, r"\1[SEP]", text)
        # print("sub text: ", text)
    except Exception as e:
        print("input: ", text)
        raise e
    text = re.sub(ch_seg_pattern, r"\1[SEP]", text)
    # print("sub ch text: ", text)
    text_list = text.split("[SEP]")
    text_list = list(filter(lambda x: len(x) != 0, text_list))
    return text_list


def load_stopwords(stopwords_path):
    stopwords_dict = {}
    with open(stopwords_path, "r") as rf:
        for line in rf:
            line = line.strip()
            if line not in stopwords_dict:
                stopwords_dict[line] = 0
            else:
                pass
    return stopwords_dict


def text_process(text, max_length):
    """分割文本"""
    texts = text_segmentate(text)

    result, length = [], 0
    for text in texts:
        if length + len(text) > max_length * 1.3 and len(result) >= 3:
            yield result
            result, length = [], 0
        result.append(text)
        length += len(text)
    if result and len(result) >= 3:
        yield result


def text_process_split_long_content(text, max_length):
    """分割长文本"""
    texts = text_segmentate(text)

    result, sentence_num = "", 0
    for text in texts:
        if len(text) > 500:
            if len(result) > 300 and sentence_num >= 3:
                yield result
                result, sentence_num = "", 0
            else:
                result, sentence_num = "", 0
                continue
        else:
            if len(result) + len(text) > max_length * 1.1 and sentence_num >= 3:
                yield result
                result, sentence_num = "", 0
            result += text
            sentence_num += 1

    if result and sentence_num >= 3:
        yield result


def gather_join(texts, idxs):
    """取出对应的text，然后拼接起来"""
    return "".join([texts[i] for i in idxs])


def gather_join_f1(texts_token, idsx):
    join_texts = []
    for id in idsx:
        join_texts.extend(texts_token[id])
    return join_texts


def compute_rouge(source, target):
    """计算rouge-1、rouge-2、rouge-l"""
    source, target = " ".join(source), " ".join(target)
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            "rouge-1": scores[0]["rouge-1"]["f"],
            "rouge-2": scores[0]["rouge-2"]["f"],
            "rouge-l": scores[0]["rouge-l"]["f"],
        }
    except ValueError:
        return {
            "rouge-1": 0.0,
            "rouge-2": 0.0,
            "rouge-l": 0.0,
        }


def remove_stopwords(texts, stopwords_dict):
    for i, text in enumerate(texts):
        texts[i] = list(filter(lambda x: x not in stopwords_dict, text))
    return texts


def pseudo_summary_f1(texts, stopwords, tokenizer, max_length, rouge_strategy="rouge-l"):
    """构建伪标签摘要数据集"""
    summary_rate = 0.25
    max_length = max_length - 1
    texts_tokens = []
    sentece_idxs_vec = []
    for text in texts:
        if len(texts) == 0:
            continue
        try:
            ids = tokenizer.encode(text.strip())[:-1]
        except ValueError:
            print("error, input : ", text)
            raise ValueError
        sentece_idxs_vec.append(ids)
        tokens = [tokenizer._convert_id_to_token(token) for token in ids]
        texts_tokens.append(tokens)

    texts_tokens_rm = remove_stopwords(texts_tokens, stopwords)
    source_idxs, target_idxs = list(range(len(texts))), []

    assert len(texts_tokens) == len(texts)
    # truncate_index = 0
    while True:
        sims = []
        for i in source_idxs:
            new_source_idxs = [j for j in source_idxs if j != i]
            new_target_idxs = sorted(target_idxs + [i])
            new_source = gather_join_f1(texts_tokens_rm, new_source_idxs)
            new_target = gather_join_f1(texts_tokens_rm, new_target_idxs)
            sim = compute_rouge(new_source, new_target)[rouge_strategy]
            sims.append(sim)
        new_idx = source_idxs[np.argmax(sims)]
        del sims
        source_idxs.remove(new_idx)
        target_idxs = sorted(target_idxs + [new_idx])
        source = gather_join(texts, source_idxs)
        target = gather_join(texts, target_idxs)
        try:
            if len(source_idxs) == 1 or 1.0 * len(target) / len(source) > summary_rate:
                break
        except ZeroDivisionError as e:
            print(e.meesage)
            print(texts)
            print("source: ", source)
            print("target: ", target)

    if len(source) < len(target):
        source, target = target, source
        source_idxs, target_idxs = target_idxs, source_idxs

    return sentece_idxs_vec, source, target, source_idxs, target_idxs


def get_input_mask(sentence_id_vec, indexs):
    target_idxs = []
    input_idxs = []
    kMaskSentenceTokenId = 2
    kEosTokenId = 1
    mask_sentence_options_cumulative_prob = [0.9, 0.9, 1, 1]
    for index in indexs:
        target_idxs.extend(sentence_id_vec[index])
        choice = random.uniform(0, 1)
        if choice < mask_sentence_options_cumulative_prob[0]:
            # print("mask index: ", index)
            sentence_id_vec[index] = [kMaskSentenceTokenId]
        elif choice < mask_sentence_options_cumulative_prob[1]:
            # print("replace index: ", index)
            replace_id = random.randint(0, len(sentence_id_vec))
            sentence_id_vec[index] = sentence_id_vec[replace_id]
        elif choice < mask_sentence_options_cumulative_prob[2]:
            pass
        else:
            sentence_id_vec[index] = []

    target_idxs.append(kEosTokenId)
    # print(sentence_id_vec)
    for index, sentence_id in enumerate(sentence_id_vec):
        # print(index, sentence_id)
        if len(sentence_id) == 0:
            continue
        input_idxs.extend(sentence_id_vec[index])

    input_idxs.append(kEosTokenId)
    return input_idxs, target_idxs


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def padding_to_maxlength(ids, max_length, pad_id):
    cur_len = len(ids)
    len_diff = max_length - cur_len
    return ids + [pad_id] * len_diff, [1] * cur_len + [0] * len_diff
