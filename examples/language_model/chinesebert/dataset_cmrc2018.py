#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from functools import partial

from paddle.io import BatchSampler, DataLoader
from paddlenlp.data import Dict, Pad, Stack
from paddlenlp.datasets import load_dataset

from utils import load_pickle, save_pickle


# this right
def prepare_train_features_paddlenlp(examples, tokenizer, args):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
    # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
    contexts = [examples[i]["context"] for i in range(len(examples))]
    questions = [examples[i]["question"] for i in range(len(examples))]

    tokenized_examples = tokenizer(questions,
                                   contexts,
                                   stride=args.doc_stride,
                                   max_seq_len=args.max_seq_length)

    # Let's label those examples!
    for i, tokenized_example in enumerate(tokenized_examples):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_example["input_ids"]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offsets = tokenized_example["offset_mapping"]

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example["token_type_ids"]

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = tokenized_example["overflow_to_sample"]
        answers = examples[sample_index]["answers"]
        answer_starts = examples[sample_index]["answer_starts"]

        # If no answers are given, set the cls_index as answer.
        if len(answer_starts) == 0:
            tokenized_examples[i]["start_positions"] = cls_index
            tokenized_examples[i]["end_positions"] = cls_index
        else:
            # Start/end character index of the answer in the text.
            start_char = answer_starts[0]
            end_char = start_char + len(answers[0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            # Minus one more to reach actual text
            token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char):
                tokenized_examples[i]["start_positions"] = cls_index
                tokenized_examples[i]["end_positions"] = cls_index
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while (token_start_index < len(offsets)
                       and offsets[token_start_index][0] <= start_char):
                    token_start_index += 1
                tokenized_examples[i]["start_positions"] = token_start_index - 1
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples[i]["end_positions"] = token_end_index + 1

    return tokenized_examples


# this right
def prepare_dev_features_paddlenlp(examples, tokenizer, args):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
    # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
    contexts = [examples[i]["context"] for i in range(len(examples))]
    questions = [examples[i]["question"] for i in range(len(examples))]

    tokenized_examples = tokenizer(questions,
                                   contexts,
                                   stride=args.doc_stride,
                                   max_seq_len=args.max_seq_length)

    # For validation, there is no need to compute start and end positions
    for i, tokenized_example in enumerate(tokenized_examples):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example["token_type_ids"]

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = tokenized_example["overflow_to_sample"]
        tokenized_examples[i]["example_id"] = examples[sample_index]["id"]

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples[i]["offset_mapping"] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_example["offset_mapping"])
        ]

    return tokenized_examples


def get_train_dataloader(tokenizer, args):
    splits = "train"
    data_dir = args.data_dir
    filename = os.path.join(data_dir, "cmrc2018_" + splits + ".pkl")

    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("cmrc2018", splits=splits)
        ds.map(
            partial(prepare_train_features_paddlenlp,
                    tokenizer=tokenizer,
                    args=args),
            batched=True,
            lazy=False,
        )
        save_pickle(ds, filename)

    batch_sampler = BatchSampler(ds,
                                 batch_size=args.train_batch_size,
                                 shuffle=True)

    batchify_fn = lambda samples, fn=Dict(
        {
            "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
            "token_type_ids": Pad(axis=0, pad_val=0),
            "pinyin_ids": Pad(axis=0, pad_val=0),
            "start_positions": Stack(dtype="int64"),
            "end_positions": Stack(dtype="int64"),
        }): fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    return data_loader


def get_dev_dataloader(tokenizer, args, splits="dev"):

    splits = "train"
    data_dir = args.data_dir
    filename = os.path.join(data_dir, "cmrc2018_" + splits + ".pkl")
    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("cmrc2018", splits=splits)
        ds.map(
            partial(prepare_dev_features_paddlenlp,
                    tokenizer=tokenizer,
                    args=args),
            batched=True,
            lazy=False,
        )
        save_pickle(ds, filename)

    batch_sampler = BatchSampler(ds,
                                 batch_size=args.eval_batch_size,
                                 shuffle=False)

    batchify_fn = lambda samples, fn=Dict(
        {
            "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
            "token_type_ids": Pad(axis=0, pad_val=0),
            "pinyin_ids": Pad(axis=0, pad_val=0),
        }): fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    return data_loader
