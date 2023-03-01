# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os

from paddlenlp.datasets import load_dataset


def cnnmail_reader(data_file):
    def punctuation_standardization(string: str):
        punctuation_dict = {"\u201c": '"', "\u201d": '"', "\u2019": "'", "\u2018": "'", "\u2013": "-"}
        for key, value in punctuation_dict.items():
            string = string.replace(key, value)
        return string

    def detokenize(string, is_target=False):
        _tok_dict = {"(": "-LRB-", ")": "-RRB-", "[": "-LSB-", "]": "-RSB-", "{": "-LCB-", "}": "-RCB-"}
        if not is_target:
            string = string.replace("<S_SEP>", "")
        else:
            string = string.replace("<S_SEP>", "[SEP]")
        for key, value in _tok_dict.items():
            string = string.replace(value, key)
        string = string.replace("''", '"')
        string = string.replace("``", '"')
        string = string.replace("`", "'")
        string = string.replace(" n't", "n't")
        string = string.replace(" 's", "'s")
        string = string.replace(" 'd", "'d")
        string = string.replace(" 'll", "'ll")
        return string

    source_texts, target_texts = [], []
    with open(f"{data_file}.source", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            line = punctuation_standardization(line)
            line = detokenize(line)
            source_texts.append(line)
    with open(f"{data_file}.target", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            line = punctuation_standardization(line)
            line = detokenize(line)
            target_texts.append(line)
    assert len(source_texts) == len(target_texts)
    example_list = []
    for idx, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
        if (idx + 1) % 20000 == 0:
            print(f"Processed {idx + 1} examples")
        example = {"text_a": source_text, "text_b": target_text}
        example_list.append(example)
    return example_list


def load_local_dataset(data_path, splits):
    dataset = []
    for split in splits:
        data_file = os.path.join(data_path, split)
        dataset.append(load_dataset(cnnmail_reader, data_file=data_file, lazy=False))
    return dataset


def cnn_dm_convert_example(example, tokenizer, data_args, is_test=True):
    prompt = [tokenizer.convert_tokens_to_ids(x) for x in ("[CLS]", "[sMASK]", "ĠContent", ":")]
    source_tokens = tokenizer.encode(" " + example["text_a"], add_special_tokens=False)["input_ids"]
    if len(source_tokens) > data_args.src_length - len(prompt):
        source_tokens = source_tokens[: data_args.src_length - len(prompt)]
    source_tokens = prompt + source_tokens
    if len(source_tokens) < data_args.src_length:
        source_tokens = source_tokens + [tokenizer.pad_token_id] * (data_args.src_length - len(source_tokens))
    sep = len(source_tokens)
    position_ids = list(range(len(source_tokens)))
    block_position_ids = [0] * len(source_tokens)
    mask_position = source_tokens.index(tokenizer.smask_token_id)
    if not is_test:
        target_tokens = tokenizer.encode(" " + example["text_b"], add_special_tokens=False)["input_ids"]
        target_tokens = target_tokens + [tokenizer.eop_token_id]
        if len(target_tokens) > data_args.tgt_length:
            target_tokens = target_tokens[: data_args.tgt_length]
        loss_mask = [1] * len(target_tokens)
        if len(target_tokens) < data_args.tgt_length:
            loss_mask = loss_mask + [0] * (data_args.tgt_length - len(target_tokens))
            target_tokens = target_tokens + [tokenizer.pad_token_id] * (data_args.tgt_length - len(target_tokens))
        tokens = source_tokens + [tokenizer.sop_token_id] + target_tokens[:-1]
        loss_mask = [0] * len(source_tokens) + loss_mask
        target_ids = [0] * len(source_tokens) + target_tokens
        position_ids = position_ids + [mask_position] * len(target_tokens)
        block_position_ids = block_position_ids + list(range(1, len(target_tokens) + 1))
        position_ids = [position_ids, block_position_ids]
        example = {
            "input_ids": tokens,
            "labels": target_ids,
            "attention_mask": sep,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }
    else:
        target_tokens = tokenizer.encode(" " + example["text_b"], add_special_tokens=False)["input_ids"]
        target_tokens = target_tokens + [tokenizer.eop_token_id]
        if len(target_tokens) > data_args.tgt_length:
            target_tokens = target_tokens[: data_args.tgt_length]
        if len(target_tokens) < data_args.tgt_length:
            target_tokens = target_tokens + [tokenizer.pad_token_id] * (data_args.tgt_length - len(target_tokens))
        labels = [0] * len(source_tokens) + target_tokens

        tokens = source_tokens + [tokenizer.sop_token_id]
        position_ids = position_ids + [mask_position]
        block_position_ids = block_position_ids + [1]
        position_ids = [position_ids, block_position_ids]

        example = {
            "input_ids": tokens,
            "attention_mask": sep,
            "position_ids": position_ids,
            "labels": labels,
        }
    return example
