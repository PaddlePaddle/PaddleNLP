# -*- coding: UTF-8 -*-
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
import numpy as np

import paddle
from paddlenlp.utils.log import logger


def create_dataloader(dataset_origin,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset_origin.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(dataset,
                                                          batch_size=batch_size,
                                                          shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)
    return paddle.io.DataLoader(dataset=dataset,
                                batch_sampler=batch_sampler,
                                collate_fn=batchify_fn,
                                return_list=True)


#TODO
def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    """
    Args:
        example(obj:`list(str)`): The list of text to be converted to ids.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        p_embedding_num(obj:`int`) The number of p-embedding.
    Returns:
        input_ids(obj:`list[int]`): The list of query token ids.
        token_type_ids(obj: `list[int]`): List of query sequence pair mask.
        mask_positions(obj: `list[int]`): The list of mask_positions.
        mask_lm_labels(obj: `list[int]`): The list of mask_lm_labels.
    """

    # Replace <unk> with '[MASK]'

    # Step1: gen mask ids
    if is_test:
        label_length = example["label_length"]
    else:
        text_label = example["text_label"]
        label_length = len(text_label)

    mask_tokens = ["[MASK]"] * label_length
    mask_ids = tokenizer.convert_tokens_to_ids(mask_tokens)

    sentence1 = example["sentence1"]
    if "<unk>" in sentence1:
        start_mask_position = sentence1.index("<unk>") + 1
        sentence1 = sentence1.replace("<unk>", "")
        encoded_inputs = tokenizer(text=sentence1, max_seq_len=max_seq_length)
        src_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]

        # Step2: Insert "[MASK]" to src_ids based on start_mask_position
        src_ids = src_ids[0:start_mask_position] + mask_ids + src_ids[
            start_mask_position:]
        token_type_ids = token_type_ids[0:start_mask_position] + [0] * len(
            mask_ids) + token_type_ids[start_mask_position:]

        # calculate mask_positions
        mask_positions = [
            index + start_mask_position for index in range(label_length)
        ]
    else:
        sentence2 = example['sentence2']
        start_mask_position = sentence2.index("<unk>") + 1
        sentence2 = sentence2.replace("<unk>", "")

        encoded_inputs = tokenizer(text=sentence2, max_seq_len=max_seq_length)
        src_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]
        src_ids = src_ids[0:start_mask_position] + mask_ids + src_ids[
            start_mask_position:]
        token_type_ids = token_type_ids[0:start_mask_position] + [0] * len(
            mask_ids) + token_type_ids[start_mask_position:]

        encoded_inputs = tokenizer(text=sentence1, max_seq_len=max_seq_length)
        sentence1_src_ids = encoded_inputs["input_ids"][1:]
        src_ids = sentence1_src_ids + src_ids
        token_type_ids += [1] * len(src_ids)
        mask_positions = [
            index + start_mask_position + len(sentence1)
            for index in range(label_length)
        ]

    token_type_ids = [0] * len(src_ids)

    assert len(src_ids) == len(
        token_type_ids), "length src_ids, token_type_ids must be equal"

    length = len(src_ids)
    if length > 512:
        src_ids = src_ids[:512]
        token_type_ids = token_type_ids[:512]

    if is_test:
        return src_ids, token_type_ids, mask_positions
    else:
        mask_lm_labels = tokenizer(
            text=text_label, max_seq_len=max_seq_length)["input_ids"][1:-1]

        assert len(mask_lm_labels) == len(
            mask_positions
        ) == label_length, "length of mask_lm_labels:{} mask_positions:{} label_length:{} not equal".format(
            mask_lm_labels, mask_positions, text_label)

        return src_ids, token_type_ids, mask_positions, mask_lm_labels


def convert_chid_example(example, tokenizer, max_seq_length=512, is_test=False):
    """
    Args:
        example(obj:`list(str)`): The list of text to be converted to ids.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        p_embedding_num(obj:`int`) The number of p-embedding.
    Returns:
        input_ids(obj:`list[int]`): The list of query token ids.
        token_type_ids(obj: `list[int]`): List of query sequence pair mask.
        mask_positions(obj: `list[int]`): The list of mask_positions.
        mask_lm_labels(obj: `list[int]`): The list of mask_lm_labels.
        mask_lm_labels(obj: `list[int]`): The list of mask_lm_labels.
    """
    # FewClue Task `Chid`' label's position must be calculated by special token: "淠"

    seg_tokens = tokenizer.tokenize(example["sentence1"])

    # find insert position of `[MASK]`
    start_mask_position = seg_tokens.index("淠") + 1
    seg_tokens.remove("淠")

    sentence1 = "".join(seg_tokens)
    candidates = example["candidates"]
    candidate_labels_ids = [
        tokenizer(text=idom)["input_ids"][1:-1] for idom in candidates
    ]

    sentence1 = example["sentence1"]

    encoded_inputs = tokenizer(text=sentence1, max_seq_len=max_seq_length)
    src_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    # Step1: gen mask ids
    if is_test:
        label_length = example["label_length"]
    else:
        text_label = example["text_label"]
        label_length = len(text_label)

    mask_tokens = ["[MASK]"] * label_length
    mask_ids = tokenizer.convert_tokens_to_ids(mask_tokens)

    # Step2: Insert "[MASK]" to src_ids based on start_mask_position
    src_ids = src_ids[0:start_mask_position] + mask_ids + src_ids[
        start_mask_position:]
    token_type_ids = token_type_ids[0:start_mask_position] + [0] * len(
        mask_ids) + token_type_ids[start_mask_position:]

    # calculate mask_positions
    mask_positions = [
        index + start_mask_position for index in range(label_length)
    ]

    # token_type_ids = [0] * len(src_ids)

    assert len(src_ids) == len(
        token_type_ids), "length src_ids, token_type_ids must be equal"

    length = len(src_ids)
    if length > 512:
        src_ids = src_ids[:512]
        token_type_ids = token_type_ids[:512]

    if is_test:
        return src_ids, token_type_ids, mask_positions, candidate_labels_ids
    else:
        mask_lm_labels = tokenizer(
            text=text_label, max_seq_len=max_seq_length)["input_ids"][1:-1]

        assert len(mask_lm_labels) == len(
            mask_positions
        ) == label_length, "length of mask_lm_labels:{} mask_positions:{} label_length:{} not equal".format(
            mask_lm_labels, mask_positions, text_label)

        return src_ids, token_type_ids, mask_positions, mask_lm_labels, candidate_labels_ids


def transform_iflytek(example,
                      label_normalize_dict=None,
                      is_test=False,
                      pattern_id=0):

    if is_test:
        # When do_test, set label_length field to point
        # where to insert [MASK] id
        example["label_length"] = 2

        if pattern_id == 0:
            example["sentence1"] = u'作为一款<unk>应用，' + example["sentence"]
        elif pattern_id == 1:
            example["sentence1"] = u'这是一款<unk>应用！' + example["sentence"]
        elif pattern_id == 2:
            example["sentence1"] = example["sentence"] + u'。 和<unk>有关'
        elif pattern_id == 3:
            example["sentence1"] = example["sentence"] + u'。 这是<unk>!'
        del example["sentence"]

        return example
    else:
        origin_label = example['label_des']

        # Normalize some of the labels, eg. English -> Chinese
        if origin_label in label_normalize_dict:
            example['label_des'] = label_normalize_dict[origin_label]
        else:
            # Note: Ideal way is drop these examples
            # which maybe need to change MapDataset
            # Now hard code may hurt performance of `iflytek` dataset
            example['label_des'] = "旅游"

        example["text_label"] = example["label_des"]

        if pattern_id == 0:
            example["sentence1"] = u'作为一款<unk>应用，' + example["sentence"]
        elif pattern_id == 1:
            example["sentence1"] = u'这是一款<unk>应用！' + example["sentence"]
        elif pattern_id == 2:
            example["sentence1"] = example["sentence"] + u'。 和<unk>有关'
        elif pattern_id == 3:
            example["sentence1"] = example["sentence"] + u'。 这是<unk>!'

        del example["sentence"]
        del example["label_des"]

        return example


def transform_tnews(example,
                    label_normalize_dict=None,
                    is_test=False,
                    pattern_id=0):
    if is_test:
        example["label_length"] = 2

        if pattern_id == 0:
            example["sentence1"] = example["sentence"] + u'。 这是<unk>的内容！'
        elif pattern_id == 1:
            example["sentence1"] = example["sentence"] + u'。 这是<unk>！'
        elif pattern_id == 2:
            example["sentence1"] = example["sentence"] + u'。 包含了<unk>的内容'
        elif pattern_id == 3:
            example["sentence1"] = example["sentence"] + u'。 综合来讲是<unk>的内容！'
        del example["sentence"]
        return example
    else:
        origin_label = example['label_desc']
        # Normalize some of the labels, eg. English -> Chinese
        example['label_desc'] = label_normalize_dict[origin_label]

        if pattern_id == 0:
            example["sentence1"] = example["sentence"] + u'。 这是<unk>的内容！'
        elif pattern_id == 1:
            example["sentence1"] = example["sentence"] + u'。 这是<unk>！'
        elif pattern_id == 2:
            example["sentence1"] = example["sentence"] + u'。 包含了<unk>的内容'
        elif pattern_id == 3:
            example["sentence1"] = example["sentence"] + u'。 综合来讲是<unk>的内容！'

        example["text_label"] = example["label_desc"]

        del example["sentence"]
        del example["label_desc"]

        return example


def transform_eprstmt(example,
                      label_normalize_dict=None,
                      is_test=False,
                      pattern_id=0):
    if is_test:
        example["label_length"] = 1

        if pattern_id == 0:
            example["sentence1"] = u'感觉很<unk>！' + example["sentence"]
        elif pattern_id == 1:
            example["sentence1"] = u'综合来讲很<unk>！，' + example["sentence"]
        elif pattern_id == 2:
            example["sentence1"] = example["sentence"] + u'感觉非常<unk>'
        elif pattern_id == 3:
            example["sentence1"] = example["sentence"] + u'， 我感到非常<unk>'

        return example
    else:
        origin_label = example["label"]
        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]

        if pattern_id == 0:
            example["sentence1"] = u'感觉很<unk>！' + example["sentence"]
        elif pattern_id == 1:
            example["sentence1"] = u'综合来讲很<unk>！，' + example["sentence"]
        elif pattern_id == 2:
            example["sentence1"] = example["sentence"] + u'感觉非常<unk>'
        elif pattern_id == 3:
            example["sentence1"] = example["sentence"] + u'， 我感到非常<unk>'

        del example["sentence"]
        del example["label"]

        return example


def transform_ocnli(example,
                    label_normalize_dict=None,
                    is_test=False,
                    pattern_id=0):
    if is_test:
        example["label_length"] = 2
        if pattern_id == 0:
            example['sentence1'] = example['sentence1'] + "， <unk>"
        elif pattern_id == 1:
            example["sentence2"] = "和" + example['sentence2'] + u"？看来<unk>一句话"
        elif pattern_id == 2:
            example["sentence1"] = "和" + example['sentence2'] + u"？<unk>一样"
        elif pattern_id == 3:
            example["sentence2"] = "和" + example['sentence2'] + u"？<unk>一句话"

        return example
    else:
        origin_label = example["label"]
        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]
        if pattern_id == 0:
            example['sentence1'] = example['sentence1'] + "， <unk>"
        elif pattern_id == 1:
            example["sentence2"] = "和" + example['sentence2'] + u"？看来<unk>一句话"
        elif pattern_id == 2:
            example["sentence1"] = "和" + example['sentence2'] + u"？<unk>一样"
        elif pattern_id == 3:
            example["sentence2"] = "和" + example['sentence2'] + u"？<unk>一句话"

        del example["label"]

        return example


def transform_csl(example,
                  label_normalize_dict=None,
                  is_test=False,
                  pattern_id=0):
    if is_test:
        example["label_length"] = 1

        if pattern_id == 0:
            example["sentence1"] = u"本文的关键词<unk>是:" + "，".join(
                example["keyword"]) + example["abst"]
        elif pattern_id == 1:
            example["sentence1"] = example[
                "abst"] + u"。本文的关键词<unk>是:" + "，".join(example["keyword"])
        elif pattern_id == 2:
            example["sentence1"] = u"本文的内容<unk>是:" + "，".join(
                example["keyword"]) + example["abst"]
        elif pattern_id == 3:
            example["sentence1"] = example[
                "abst"] + u"。本文的内容<unk>是:" + "，".join(example["keyword"])

        del example["abst"]
        del example["keyword"]

        return example
    else:
        origin_label = example["label"]
        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]

        if pattern_id == 0:
            example["sentence1"] = u"本文的关键词<unk>是:" + "，".join(
                example["keyword"]) + example["abst"]
        elif pattern_id == 1:
            example["sentence1"] = example[
                "abst"] + u"。本文的关键词<unk>是:" + "，".join(example["keyword"])
        elif pattern_id == 2:
            example["sentence1"] = u"本文的内容<unk>是:" + "，".join(
                example["keyword"]) + example["abst"]
        elif pattern_id == 3:
            example["sentence1"] = example[
                "abst"] + u"。本文的内容<unk>是:" + "，".join(example["keyword"])

        del example["label"]
        del example["abst"]
        del example["keyword"]

        return example


def transform_csldcp(example,
                     label_normalize_dict=None,
                     is_test=False,
                     pattern_id=0):
    if is_test:
        example["label_length"] = 2

        if pattern_id == 0:
            example["sentence1"] = u'这篇关于<unk>的文章讲了' + example["content"]
        elif pattern_id == 1:
            example["sentence1"] = example["content"] + u'和<unk>息息相关'
        elif pattern_id == 2:
            example["sentence1"] = u'这是一篇和<unk>息息相关的文章' + example["content"]
        elif pattern_id == 3:
            example["sentence1"] = u'很多很多<unk>的文章！' + example["content"]

        del example["content"]
        return example
    else:
        origin_label = example["label"]
        # Normalize some of the labels, eg. English -> Chinese
        normalized_label = label_normalize_dict[origin_label]
        example['text_label'] = normalized_label
        if pattern_id == 0:
            example["sentence1"] = u'这篇关于<unk>的文章讲了' + example["content"]
        elif pattern_id == 1:
            example["sentence1"] = example["content"] + u'和<unk>息息相关'
        elif pattern_id == 2:
            example["sentence1"] = u'这是一篇和<unk>息息相关的文章' + example["content"]
        elif pattern_id == 3:
            example["sentence1"] = u'很多很多<unk>的文章！' + example["content"]

        del example["label"]
        del example["content"]

        return example


def transform_bustm(example,
                    label_normalize_dict=None,
                    is_test=False,
                    pattern_id=0):
    if is_test:
        # Label: ["很"， "不"]
        example["label_length"] = 1
        if pattern_id == 0:
            example['sentence1'] = "<unk>是一句话. " + example['sentence1'] + "，"
        elif pattern_id == 1:
            example['sentence2'] = "，" + example['sentence2'] + "。<unk>是一句话. "
        elif pattern_id == 2:
            example['sentence1'] = "讲的<unk>是一句话。" + example['sentence1'] + "，"
        elif pattern_id == 3:
            example['sentence1'] = "，" + example['sentence2'] + "。讲的<unk>是一句话. "

        return example
    else:
        origin_label = str(example["label"])

        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]
        if pattern_id == 0:
            example['sentence1'] = "<unk>是一句话. " + example['sentence1'] + "，"
        elif pattern_id == 1:
            example['sentence2'] = "，" + example['sentence2'] + "。<unk>是一句话. "
        elif pattern_id == 2:
            example['sentence1'] = "讲的<unk>是一句话。" + example['sentence1'] + "，"
        elif pattern_id == 3:
            example['sentence1'] = "，" + example['sentence2'] + "。讲的<unk>是一句话. "

        del example["label"]

        return example


def transform_chid(example,
                   label_normalize_dict=None,
                   is_test=False,
                   pattern_id=0):

    if is_test:
        example["label_length"] = 4
        example["sentence1"] = example["content"].replace("#idiom#", "淠")
        del example["content"]

        return example
    else:
        label_index = int(example['answer'])
        candidates = example["candidates"]
        example["text_label"] = candidates[label_index]

        # Note: `#idom#` represent a idom which must be replaced with rarely-used Chinese characters
        # to get the label's position after the text processed by tokenizer
        #ernie
        example["sentence1"] = example["content"].replace("#idiom#", "淠")
        del example["content"]

        return example


def transform_cluewsc(example,
                      label_normalize_dict=None,
                      is_test=False,
                      pattern_id=0):
    if is_test:
        example["label_length"] = 2
        text = example["text"]
        span1_text = example["target"]["span1_text"]
        span2_text = example["target"]["span2_text"]

        # example["sentence1"] = text.replace(span2_text, span1_text)
        if pattern_id == 0:
            example["sentence1"] = text + span2_text + "<unk>地指代" + span1_text
        elif pattern_id == 1:
            example["sentence1"] = text + span2_text + "<unk>地意味着" + span1_text
        elif pattern_id == 2:
            example["sentence1"] = text + span2_text + "<unk>地代表" + span1_text
        elif pattern_id == 3:
            example["sentence1"] = text + span2_text + "<unk>地表示了" + span1_text
        del example["text"]
        # del example["target"]

        return example
    else:
        origin_label = example["label"]
        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]
        # example['text_label'] = origin_label
        text = example["text"]
        span1_text = example["target"]["span1_text"]
        span2_text = example["target"]["span2_text"]

        # example["sentence1"] = text.replace(span2_text, span1_text)
        if pattern_id == 0:
            example["sentence1"] = text + span2_text + "<unk>地指代" + span1_text
        elif pattern_id == 1:
            example["sentence1"] = text + span2_text + "<unk>地意味着" + span1_text
        elif pattern_id == 2:
            example["sentence1"] = text + span2_text + "<unk>地代表" + span1_text
        elif pattern_id == 3:
            example["sentence1"] = text + span2_text + "<unk>地表示了" + span1_text

        del example["label"]
        del example["text"]
        del example["target"]

        return example


transform_fn_dict = {
    "iflytek": transform_iflytek,
    "tnews": transform_tnews,
    "eprstmt": transform_eprstmt,
    "bustm": transform_bustm,
    "ocnli": transform_ocnli,
    "csl": transform_csl,
    "csldcp": transform_csldcp,
    "cluewsc": transform_cluewsc,
    "chid": transform_chid
}
