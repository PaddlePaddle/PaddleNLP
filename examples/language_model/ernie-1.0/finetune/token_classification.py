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

import argparse
import os
import random
import time
import math
import sys
from functools import partial

import numpy as np
import paddle
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.utils.log import logger

# from paddlenlp.trainer.trainer_base import TrainerBase
sys.path.insert(0, os.path.abspath("."))
from utils import Dict


def tokenize_and_align_labels(example, tokenizer, no_entity_id,
                              max_seq_len=512):
    labels = example['labels']
    example = example['tokens']
    tokenized_input = tokenizer(
        example,
        is_split_into_words=True,
        max_seq_len=max_seq_len, )

    # -2 for [CLS] and [SEP]
    if len(tokenized_input['input_ids']) - 2 < len(labels):
        labels = labels[:len(tokenized_input['input_ids']) - 2]
    tokenized_input['labels'] = [no_entity_id] + labels + [no_entity_id]
    tokenized_input['labels'] += [no_entity_id] * (
        len(tokenized_input['input_ids']) - len(tokenized_input['labels']))

    return tokenized_input


def ner_collator(tokenizer, args):
    batchify_fn = lambda samples, fn=Dict({
        'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int32'),  # input
        'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int32'),  # segment
        'labels': Pad(axis=0, pad_val=args.ignore_label, dtype='int64')  # label
    }): fn(samples)

    return batchify_fn


def ner_trans_fn(example, tokenizer, args):
    return tokenize_and_align_labels(
        example,
        tokenizer=tokenizer,
        no_entity_id=args.no_entity_id,
        max_seq_len=args.max_seq_length)
