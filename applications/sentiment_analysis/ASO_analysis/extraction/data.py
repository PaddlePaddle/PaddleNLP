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


def load_dict(dict_path):
    with open(dict_path, "r", encoding="utf-8") as f:
        words = [word.strip() for word in f.readlines()]
        word2id = dict(zip(words, range(len(words))))
        id2word = dict((v, k) for k, v in word2id.items())

        return word2id, id2word


def convert_example_to_feature(example, tokenizer, label2id, max_seq_len=512, is_test=False):
    example = example["text"].rstrip().split("\t")
    text = list(example[0])
    if not is_test:
        label = example[1].split(" ")
        assert len(text) == len(label)
        new_text = []
        new_label = []
        for text_ch, label_ch in zip(text, label):
            if text_ch.strip():
                new_text.append(text_ch)
                new_label.append(label_ch)
        new_label = (
            [label2id["O"]] + [label2id[label_term] for label_term in new_label][: (max_seq_len - 2)] + [label2id["O"]]
        )
        encoded_inputs = tokenizer(new_text, is_split_into_words="token", max_seq_len=max_seq_len, return_length=True)
        encoded_inputs["labels"] = new_label
        assert len(encoded_inputs["input_ids"]) == len(
            new_label
        ), f"input_ids: {len(encoded_inputs['input_ids'])}, label: {len(new_label)}"
    else:
        new_text = [text_ch for text_ch in text if text_ch.strip()]
        encoded_inputs = tokenizer(new_text, is_split_into_words="token", max_seq_len=max_seq_len, return_length=True)

    return encoded_inputs
