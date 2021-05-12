# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import json

import paddle

from paddlenlp.utils.log import logger


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)


def convert_example(example, tokenizer, max_seq_length=512, p_embedding_num=5):
    """
    Builds model inputs from a sequence.
        
    A BERT sequence has the following format:

    - single sequence: ``[CLS] [MASK] [unused1] [unused2] ... [unusedN] X [SEP]``

    Args:
        example(obj:`list(str)`): The list of text to be converted to ids.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of query token ids.
        token_type_ids(obj: `list[int]`): List of query sequence pair mask.
        mask_positions(obj: `list[int]`): The list of mask_positions.
        mask_lm_labels(obj: `list[int]`): The list of mask_lm_labels.
    """

    sentence1 = example["sentence1"]

    encoded_inputs = tokenizer(text=sentence1, max_seq_len=max_seq_length)
    src_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    P_tokens = ["[unused{}]".format(i) for i in range(p_embedding_num)]
    # insert [MASK] on mask_positions
    text_label = example["text_label"]
    #print("text_label:{}".format(text_label))
    label_length = len(text_label)
    #print("label_length:{}".format(label_length))

    Mask_tokens = ["[MASK]"] * label_length

    mask_ids = tokenizer.convert_tokens_to_ids(Mask_tokens)
    p_token_ids = tokenizer.convert_tokens_to_ids(P_tokens)

    # add [MASK] at the begin of sentence, and then P-embedding
    src_ids = [src_ids[0]] + mask_ids + p_token_ids + src_ids[1:]
    mask_positions = [index + 1 for index in range(label_length)]

    mask_lm_labels = tokenizer(
        text=text_label, max_seq_len=max_seq_length)["input_ids"][1:-1]
    assert len(mask_lm_labels) == len(
        mask_positions
    ) == label_length, "length of mask_lm_labels:{} mask_positions:{} label_length:{} not equal".format(
        mask_lm_labels, mask_positions, text_label)

    token_type_ids = [0] * len(src_ids)

    if "sentence2" in example:
        encoded_inputs = tokenizer(text=sentence1, max_seq_len=max_seq_length)
        sentence2_src_ids = encoded_inputs["input_ids"][1:]
        src_ids += sentence2_src_ids
        token_type_ids += [1] * len(sentence2_src_ids)

    assert len(src_ids) == len(
        token_type_ids), "length src_ids, token_type_ids must be equal"

    return src_ids, token_type_ids, mask_positions, mask_lm_labels


def read_iflytek(data_path, label_normalize_dict=None):
    """Reads data."""
    examples = []
    with open(data_path, encoding='utf-8') as f:
        print("data_path:{}".format(data_path))
        for line in f:
            example = json.loads(line)

            origin_label = example['label_des']
            if origin_label in label_normalize_dict:
                # 针对部分 Label 进行标准化， eg. 英文 -> 中文
                normalized_label = label_normalize_dict[origin_label]
                example['label_des'] = normalized_label
            else:
                # filter illegal example
                continue

            example["sentence1"] = example["sentence"]
            example["text_label"] = example["label_des"]
            del example["sentence"]
            del example["label_des"]

            examples.append(example)

    return examples


def read_tnews(data_path, label_normalize_dict=None):
    """Reads data."""
    examples = []
    with open(data_path, encoding='utf-8') as f:
        print("data_path:{}".format(data_path))
        for line in f:
            example = json.loads(line)

            origin_label = example['label_desc']
            if origin_label in label_normalize_dict:
                # 针对部分 Label 进行标准化， eg. 英文 -> 中文
                normalized_label = label_normalize_dict[origin_label]
                example['label_desc'] = normalized_label
            else:
                # filter illegal example
                continue

            example["sentence1"] = example["sentence"]
            example["text_label"] = example["label_desc"]
            del example["sentence"]
            del example["label_desc"]

            examples.append(example)

    return examples


def read_eprstmt(data_path, label_normalize_dict=None):
    """Reads data."""
    examples = []
    with open(data_path, encoding='utf-8') as f:
        print("data_path:{}".format(data_path))
        for line in f:
            example = json.loads(line)

            origin_label = example["label"]

            if origin_label in label_normalize_dict:
                # 针对部分 Label 进行标准化， eg. 英文 -> 中文
                normalized_label = label_normalize_dict[origin_label]
                example['text_label'] = normalized_label
            else:
                # filter illegal example
                continue

            example['sentence1'] = example["sentence"]

            del example["sentence"]
            del example["label"]

            examples.append(example)

    return examples


def read_bustm(data_path, label_normalize_dict=None):
    """Reads data."""
    examples = []
    with open(data_path, encoding='utf-8') as f:
        print("data_path:{}".format(data_path))
        for line in f:
            example = json.loads(line)

            origin_label = example["label"]

            if origin_label in label_normalize_dict:
                # 针对部分 Label 进行标准化， eg. 英文 -> 中文
                normalized_label = label_normalize_dict[origin_label]
                example['text_label'] = normalized_label
            else:
                # filter illegal example
                continue

            #example["sentence1"] = "第一个句子是: " + example["sentence1"]
            #example["sentence2"] = "另一个句子是: " + example["sentence2"]

            del example["label"]

            examples.append(example)

    return examples


def read_ocnli(data_path, label_normalize_dict=None):
    """Reads data."""
    examples = []
    with open(data_path, encoding='utf-8') as f:
        print("data_path:{}".format(data_path))
        for line in f:
            example = json.loads(line)

            origin_label = example["label"]

            if origin_label in label_normalize_dict:
                # 针对部分 Label 进行标准化， eg. 英文 -> 中文
                normalized_label = label_normalize_dict[origin_label]
                example['text_label'] = normalized_label
            else:
                # filter illegal example
                continue

            del example["label"]

            examples.append(example)

    return examples


def read_csl(data_path, label_normalize_dict=None):
    """Reads data."""
    examples = []
    with open(data_path, encoding='utf-8') as f:
        print("data_path:{}".format(data_path))
        for line in f:
            example = json.loads(line)

            origin_label = example["label"]

            if origin_label in label_normalize_dict:
                # 针对部分 Label 进行标准化， eg. 英文 -> 中文
                normalized_label = label_normalize_dict[origin_label]
                example['text_label'] = normalized_label
            else:
                # filter illegal example
                continue

            example["sentence1"] = "本文的关键词是:" + "，".join(example[
                "keyword"]) + example["abst"]

            del example["label"]
            del example["abst"]
            del example["keyword"]

            examples.append(example)

    return examples


def read_csldcp(data_path, label_normalize_dict=None):
    """Reads data."""
    examples = []
    with open(data_path, encoding='utf-8') as f:
        print("data_path:{}".format(data_path))
        for line in f:
            example = json.loads(line)

            origin_label = example["label"]

            if origin_label in label_normalize_dict:
                # 针对部分 Label 进行标准化， eg. 英文 -> 中文
                normalized_label = label_normalize_dict[origin_label]
                example['text_label'] = normalized_label
            else:
                # filter illegal example
                continue

            example["sentence1"] = example["content"]

            del example["label"]
            del example["content"]

            examples.append(example)

    return examples


def read_cluewsc(data_path, label_normalize_dict=None):
    """Reads data."""
    examples = []
    with open(data_path, encoding='utf-8') as f:
        print("data_path:{}".format(data_path))
        for line in f:
            example = json.loads(line)

            origin_label = example["label"]

            if origin_label in label_normalize_dict:
                # 针对部分 Label 进行标准化， eg. 英文 -> 中文
                normalized_label = label_normalize_dict[origin_label]
                example['text_label'] = normalized_label
            else:
                # filter illegal example
                continue

            text = example["text"]
            span1_text = example["target"]["span1_text"]
            span2_text = example["target"]["span2_text"]

            example["sentence1"] = text + span2_text + "指代" + span1_text

            del example["label"]
            del example["text"]
            del example["target"]

            examples.append(example)

    return examples


read_fn_dict = {
    "iflytek": read_iflytek,
    "tnews": read_tnews,
    "eprstmt": read_eprstmt,
    "bustm": read_bustm,
    "ocnli": read_ocnli,
    "csl": read_csl,
    "csldcp": read_csldcp,
    "cluewsc": read_cluewsc
}
