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

import collections
import json
import random
from typing import List

import numpy as np
import paddle
from paddle.io import Dataset

from paddlenlp.transformers.bert.tokenizer import BertTokenizer

BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])

BiENcoderBatch = collections.namedtuple(
    "BiEncoderInput",
    [
        "questions_ids",
        "question_segments",
        "context_ids",
        "ctx_segments",
        "is_positive",
        "hard_negatives",
        "encoder_type",
    ],
)


def normalize_question(question: str) -> str:
    question = question.replace("’", "'")
    return question


def normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    if ctx_text.startswith('"'):
        ctx_text = ctx_text[1:]
    if ctx_text.endswith('"'):
        ctx_text = ctx_text[:-1]
    return ctx_text


class BiEncoderSample(object):
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]


class NQdataSetForDPR(Dataset):
    """
    class for managing dataset
    """

    def __init__(self, dataPath, query_special_suffix=None):
        super(NQdataSetForDPR, self).__init__()
        self.data = self._read_json_data(dataPath)
        self.tokenizer = BertTokenizer
        self.query_special_suffix = query_special_suffix
        self.new_data = []
        for i in range(0, self.__len__()):
            self.new_data.append(self.__getitem__(i))

    def _read_json_data(self, dataPath):
        results = []
        with open(dataPath, "r", encoding="utf-8") as f:
            print("Reading file %s" % dataPath)
            data = json.load(f)
            results.extend(data)
            print("Aggregated data size: {}".format(len(results)))
        return results

    def __getitem__(self, index):
        json_sample_data = self.data[index]
        r = BiEncoderSample()
        r.query = self._process_query(json_sample_data["question"])

        positive_ctxs = json_sample_data["positive_ctxs"]

        negative_ctxs = json_sample_data["negative_ctxs"] if "negative_ctxs" in json_sample_data else []
        hard_negative_ctxs = json_sample_data["hard_negative_ctxs"] if "hard_negative_ctxs" in json_sample_data else []

        for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs:
            if "title" not in ctx:
                ctx["title"] = None

        def create_passage(ctx):
            return BiEncoderPassage(normalize_passage(ctx["text"]), ctx["title"])

        r.positive_passages = [create_passage(ctx) for ctx in positive_ctxs]
        r.negative_passages = [create_passage(ctx) for ctx in negative_ctxs]
        r.hard_negative_passages = [create_passage(ctx) for ctx in hard_negative_ctxs]

        return r

    def _process_query(self, query):
        query = normalize_question(query)

        if self.query_special_suffix and not query.endswith(self.query_special_suffix):
            query += self.query_special_suffix

        return query

    def __len__(self):
        return len(self.data)


class DataUtil:
    """
    Class for working with datasets
    """

    def __init__(self):
        self.tensorizer = BertTensorizer()

    def create_biencoder_input(
        self,
        samples: List[BiEncoderSample],
        inserted_title,
        num_hard_negatives=0,
        num_other_negatives=0,
        shuffle=True,
        shuffle_positives=False,
        hard_neg_positives=False,
        hard_neg_fallback=True,
        query_token=None,
    ):

        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        for sample in samples:

            if shuffle and shuffle_positives:
                positive_ctxs = sample.positive_passages
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample.positive_passages[0]

            neg_ctxs = sample.negative_passages
            hard_neg_ctxs = sample.hard_negative_passages
            question = sample.query

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]

            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

            all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            hard_negative_start_idx = 1
            hard_negative_end_idx = 1 + len(hard_neg_ctxs)

            current_ctxs_len = len(ctx_tensors)

            sample_ctxs_tensors = [
                self.tensorizer.text_to_tensor(ctx.text, title=ctx.title if (inserted_title and ctx.title) else None)
                for ctx in all_ctxs
            ]

            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                i
                for i in range(
                    current_ctxs_len + hard_negative_start_idx,
                    current_ctxs_len + hard_negative_end_idx,
                )
            )
            """if query_token:
                if query_token == "[START_END]":
                    query_span = _select_span
                else:
                    question_tensors.append(self.tensorizer.text_to_tensor(" ".join([query_token, question])))
            else:"""

            question_tensors.append(self.tensorizer.text_to_tensor(question))

        ctxs_tensor = paddle.concat([paddle.reshape(ctx, [1, -1]) for ctx in ctx_tensors], axis=0)
        questions_tensor = paddle.concat([paddle.reshape(q, [1, -1]) for q in question_tensors], axis=0)

        ctx_segments = paddle.zeros_like(ctxs_tensor)
        question_segments = paddle.zeros_like(questions_tensor)

        return BiENcoderBatch(
            questions_tensor,
            question_segments,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            "question",
        )


class BertTensorizer:
    def __init__(self, pad_to_max=True, max_length=256):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(
        self,
        text: str,
        title=None,
    ):
        text = text.strip()

        if title:
            token_ids = self.tokenizer.encode(
                text,
                text_pair=title,
                max_seq_len=self.max_length,
                pad_to_max_seq_len=False,
                truncation_strategy="longest_first",
            )["input_ids"]
        else:
            token_ids = self.tokenizer.encode(
                text,
                max_seq_len=self.max_length,
                pad_to_max_seq_len=False,
                truncation_strategy="longest_first",
            )["input_ids"]

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_type_id] * (seq_len - len(token_ids))
        if len(token_ids) >= seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = 102

        return paddle.to_tensor(token_ids)
