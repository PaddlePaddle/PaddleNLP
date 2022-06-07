# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import random
import numpy as np
from seqeval.metrics.sequence_labeling import get_entities


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_dict(dict_path):
    with open(dict_path, "r", encoding="utf-8") as f:
        words = [word.strip() for word in f.readlines()]
        word2id = dict(zip(words, range(len(words))))
        id2word = dict((v, k) for k, v in word2id.items())

        return word2id, id2word


def read_test_file(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip().replace(" ", "")
            yield {"text": line}


def decoding(text, tag_seq):
    assert len(text) == len(
        tag_seq), f"text len: {len(text)}, tag_seq len: {len(tag_seq)}"

    puncs = list(",.?;!，。？；！")
    splits = [idx for idx in range(len(text)) if text[idx] in puncs]

    prev = 0
    sub_texts, sub_tag_seqs = [], []
    for i, split in enumerate(splits):
        sub_tag_seqs.append(tag_seq[prev:split])
        sub_texts.append(text[prev:split])
        prev = split
    sub_tag_seqs.append(tag_seq[prev:])
    sub_texts.append((text[prev:]))

    ents_list = []
    for sub_text, sub_tag_seq in zip(sub_texts, sub_tag_seqs):
        ents = get_entities(sub_tag_seq, suffix=False)
        ents_list.append((sub_text, ents))

    aps = []
    no_a_words = []
    for sub_tag_seq, ent_list in ents_list:
        sub_aps = []
        sub_no_a_words = []
        for ent in ent_list:
            ent_name, start, end = ent
            if ent_name == "Aspect":
                aspect = sub_tag_seq[start:end + 1]
                sub_aps.append([aspect])
                if len(sub_no_a_words) > 0:
                    sub_aps[-1].extend(sub_no_a_words)
                    sub_no_a_words.clear()
            else:
                ent_name == "Opinion"
                opinion = sub_tag_seq[start:end + 1]
                if len(sub_aps) > 0:
                    sub_aps[-1].append(opinion)
                else:
                    sub_no_a_words.append(opinion)

        if sub_aps:
            aps.extend(sub_aps)
            if len(no_a_words) > 0:
                aps[-1].extend(no_a_words)
                no_a_words.clear()
        elif sub_no_a_words:
            if len(aps) > 0:
                aps[-1].extend(sub_no_a_words)
            else:
                no_a_words.extend(sub_no_a_words)

    if no_a_words:
        no_a_words.insert(0, "None")
        aps.append(no_a_words)

    return aps


def concate_aspect_and_opinion(text, aspect, opinions):
    aspect_text = ""
    for opinion in opinions:
        if text.find(aspect) <= text.find(opinion):
            aspect_text += aspect + opinion + "，"
        else:
            aspect_text += opinion + aspect + "，"
    aspect_text = aspect_text[:-1]

    return aspect_text


def save_examples(examples, save_path, idxs):
    with open(save_path, "w", encoding="utf-8") as f:
        for idx in idxs:
            line = "\t".join(examples[idx]) + "\n"
            f.write(line)


def save_dict(dict_path, dict_type):
    if dict_type not in ["ext", "cls"]:
        raise ValueError("Only ext/cls should be accepted for dict_type.")

    with open(dict_path, "w", encoding="utf-8") as f:
        if dict_type == "ext":
            label_list = ["O", "B-Aspect", "I-Aspect", "B-Opinion", "I-Opinion"]
        else:
            label_list = ["负向", "正向"]

        for label in label_list:
            f.write(label + "\n")
