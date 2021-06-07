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
from collections import defaultdict
import random

import paddle
from paddlenlp.utils.log import logger
from paddlenlp.datasets import MapDataset


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
    #logger.debug("example:{}".format(example))
    sentence1 = example["sentence1"]
    sentence2 = example["sentence2"]
    encoded_inputs = tokenizer(
        text=sentence1,
        text_pair=sentence2,
        max_seq_len=max_seq_length,
        truncation_strategy="only_first")

    src_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = example["label"]

    #logger.debug("example:{}".format(example))
    #logger.debug("src_ids:{}".format(src_ids))
    #logger.debug("token_type_ids:{}".format(token_type_ids))

    if is_test:
        return src_ids, token_type_ids
    else:
        #logger.debug("label:{}".format(label))
        return src_ids, token_type_ids, label


def convert_chid_example(example,
                         tokenizer,
                         max_seq_length=512,
                         p_embedding_num=5,
                         is_test=False):
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

    # Step2: gen p_token_ids
    p_tokens = ["[unused{}]".format(i) for i in range(p_embedding_num)]
    p_token_ids = tokenizer.convert_tokens_to_ids(p_tokens)

    # Step3: Insert "[MASK]" to src_ids based on start_mask_position
    src_ids = src_ids[0:start_mask_position] + mask_ids + src_ids[
        start_mask_position:]

    # Stpe4: Insert P-tokens at begin of sentence
    src_ids = p_token_ids + src_ids

    # calculate mask_positions
    mask_positions = [
        index + start_mask_position + len(p_token_ids)
        for index in range(label_length)
    ]

    token_type_ids = [0] * len(src_ids)

    assert len(src_ids) == len(
        token_type_ids), "length src_ids, token_type_ids must be equal"

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


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, negative_num):
        # Random negative sample number for efl strategy
        self.neg_num = negative_num

    def get_train_datasets(self, datasets, task_label_description):
        """See base class."""
        return self._create_examples(datasets, "train", task_label_description)

    def get_dev_datasets(self, datasets, task_label_description):
        """See base class."""
        return self._create_examples(datasets, "dev", task_label_description)

    def get_test_datasets(self, datasets, task_label_description):
        """See base class."""
        return self._create_examples(datasets, "test", task_label_description)


class IflytekProcessor(DataProcessor):
    """Processor for the IFLYTEK data set (CLUE version)."""

    def _create_examples(self, datasets, phase, task_label_description):
        """Creates examples for the training and dev sets."""

        examples = []

        if phase == "train":
            for example in datasets:
                true_label = str(example["label"])
                #print("origin_example:{}".format(example))
                neg_examples = []
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence']
                    new_example["sentence2"] = label_description

                    # Todo: handle imbanlanced example, maybe hurt model performance 
                    #print("true_label:{}\tlabel:{}".format(true_label, label))
                    if true_label == label:
                        new_example["label"] = 1
                        examples.append(new_example)
                    else:
                        new_example["label"] = 0
                        neg_examples.append(new_example)
                neg_examples = random.sample(neg_examples, self.neg_num)
                #print("neg_exmaples:{}".format(neg_examples))                    
                examples.extend(neg_examples)

        elif phase == "dev":
            for example in datasets:
                #print("origin_example:{}".format(example))
                true_label = example["label"]
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence']
                    new_example["sentence2"] = label_description
                    new_example["label"] = true_label
                    examples.append(new_example)
                    #print("phase_dev new_example:{}".format(new_example))

        elif phase == "test":
            for example in datasets:
                #print("origin_example:{}".format(example))
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence']
                    new_example["sentence2"] = label_description
                    #print("phase_test new_example:{}".format(new_example))
                    examples.append(new_example)

        return MapDataset(examples)


class OcnliProcessor(DataProcessor):
    """Processor for the IFLYTEK data set (CLUE version)."""

    @classmethod
    def _create_examples(cls, datasets, phase, task_label_description):
        """Creates examples for the training and dev sets."""

        examples = []

        if phase == "train":
            for example in datasets:
                true_label = str(example["label"])
                #print("origin_example:{}".format(example))
                neg_examples = []
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence1']
                    new_example["sentence2"] = label_description + example[
                        'sentence2']
                    # Todo: handle imbanlanced example, maybe hurt model performance 
                    #print("true_label:{}\tlabel:{}".format(true_label, label))
                    if true_label == label:
                        new_example["label"] = 1
                        examples.append(new_example)
                    else:
                        new_example["label"] = 0
                        neg_examples.append(new_example)
                neg_examples = random.sample(neg_examples, self.neg_num)
                examples.extend(neg_examples)

        elif phase == "dev":
            for example in datasets:
                #print("origin_example:{}".format(example))
                true_label = example["label"]
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence1']
                    new_example["sentence2"] = label_description + example[
                        'sentence2']
                    new_example["label"] = true_label
                    examples.append(new_example)
                    #print("phase_dev new_example:{}".format(new_example))

        elif phase == "test":
            for example in datasets:
                #print("origin_example:{}".format(example))
                for label, label_description in task_label_description.items():
                    new_example = dict()
                    new_example["sentence1"] = example['sentence1']
                    new_example["sentence2"] = label_description + example[
                        'sentence2']
                    #print("phase_test new_example:{}".format(new_example))
                    examples.append(new_example)

        return MapDataset(examples)


def transform_iflytek(example, label_normalize_dict=None, phase="train"):

    if is_test:
        # When do_test, set label_length field to point
        # where to insert [MASK] id
        example["label_length"] = 2
        example["sentence1"] = example["sentence"]
        del example["sentence"]
        return example
    else:
        print(example)
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
        example["sentence1"] = example["sentence"]

        del example["sentence"]
        del example["label_des"]

        return example


def transform_tnews(example, label_normalize_dict=None, is_test=False):
    if is_test:
        example["label_length"] = 2
        example["sentence1"] = example["sentence"]
        del example["sentence"]
        return example
    else:
        origin_label = example['label_desc']
        # Normalize some of the labels, eg. English -> Chinese
        example['label_desc'] = label_normalize_dict[origin_label]

        example["sentence1"] = example["sentence"]
        example["text_label"] = example["label_desc"]

        del example["sentence"]
        del example["label_desc"]

        return example


def transform_eprstmt(example, label_normalize_dict=None, is_test=False):
    if is_test:
        example["label_length"] = 1
        example['sentence1'] = example["sentence"]
        return example
    else:
        origin_label = example["label"]
        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]
        example['sentence1'] = example["sentence"]

        del example["sentence"]
        del example["label"]

        return example


def transform_ocnli(example, label_normalize_dict=None, is_test=False):
    if is_test:
        example["label_length"] = 1
        return example
    else:
        origin_label = example["label"]
        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]

        del example["label"]

        return example


def transform_csl(example, label_normalize_dict=None, is_test=False):
    if is_test:
        example["label_length"] = 1
        example["sentence1"] = "本文的关键词是:" + "，".join(example[
            "keyword"]) + example["abst"]

        del example["abst"]
        del example["keyword"]

        return example
    else:
        origin_label = example["label"]
        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]

        example["sentence1"] = "本文的关键词是:" + "，".join(example[
            "keyword"]) + example["abst"]

        del example["label"]
        del example["abst"]
        del example["keyword"]

        return example


def transform_csldcp(example, label_normalize_dict=None, is_test=False):
    if is_test:
        example["label_length"] = 2
        example["sentence1"] = example["content"]
        del example["content"]
        return example
    else:
        origin_label = example["label"]
        # Normalize some of the labels, eg. English -> Chinese
        normalized_label = label_normalize_dict[origin_label]
        example['text_label'] = normalized_label
        example["sentence1"] = example["content"]

        del example["label"]
        del example["content"]

        return example


def transform_bustm(example, label_normalize_dict=None, is_test=False):
    if is_test:
        # Label: ["很"， "不"]
        example["label_length"] = 1
        return example
    else:
        origin_label = str(example["label"])

        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]

        del example["label"]

        return example


def transform_chid(example, label_normalize_dict=None, is_test=False):

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
        example["sentence1"] = example["content"].replace("#idiom#", "淠")
        del example["content"]

        return example


def transform_cluewsc(example, label_normalize_dict=None, is_test=False):
    if is_test:
        example["label_length"] = 2
        text = example["text"]
        span1_text = example["target"]["span1_text"]
        span2_text = example["target"]["span2_text"]
        example["sentence1"] = text + span2_text + "指代" + span1_text

        del example["text"]
        del example["target"]

        return example
    else:
        origin_label = example["label"]
        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]

        text = example["text"]
        span1_text = example["target"]["span1_text"]
        span2_text = example["target"]["span2_text"]
        example["sentence1"] = text + span2_text + "指代" + span1_text

        del example["label"]
        del example["text"]
        del example["target"]

        return example


Processor_dict = {
    "iflytek": IflytekProcessor,
    "tnews": transform_tnews,
    "eprstmt": transform_eprstmt,
    "bustm": transform_bustm,
    "ocnli": OcnliProcessor,
    "csl": transform_csl,
    "csldcp": transform_csldcp,
    "cluewsc": transform_cluewsc,
    "chid": transform_chid
}
