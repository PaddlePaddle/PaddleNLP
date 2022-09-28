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
from functools import partial
from paddle.io import BatchSampler, DataLoader
from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import load_dataset

from utils import load_pickle, save_pickle
import collections

GLUE_PROCESSED = collections.OrderedDict([
    ("cola", (["cola sentence: "], ["not_acceptable", "acceptable"])),
    ("sst-2", (["sst2 sentence: "], ["negative", "positive"])),
    (
        "mrpc",
        (["mrpc sentence1: ", " sentence2: "], ["not_equivalent",
                                                "equivalent"]),
    ),
    ("sts-b", (["stsb sentence1: ", " sentence2: "], None)),
    ("qqp", (["qqp question1: ",
              " question2: "], ["not_duplicate", "duplicate"])),
    (
        "mnli",
        (
            ["mnli hypothesis: ", " premise: "],
            ["contradiction", "entailment", "neutral"],
        ),
    ),
    (
        "qnli",
        (["qnli question: ", " sentence: "], ["entailment", "not_entailment"]),
    ),
    (
        "rte",
        (["rte sentence1: ",
          " rte sentence2: "], ["entailment", "not_entailment"]),
    ),
])


def trans_func(example, tokenizer, args):
    task_name = args.task_name
    processed, label = GLUE_PROCESSED[task_name]
    if label:
        id2label = dict(zip(range(len(label)), label))
    else:
        id2label = None

    if not args.is_test:
        if id2label:
            label_text = id2label[example["labels"]]
        else:
            label_text = str(example["labels"])
        target = tokenizer(label_text,
                           return_token_type_ids=False,
                           return_attention_mask=True)

    if len(processed) == 1:
        text = processed[0] + example["sentence"]
    else:
        text = processed[0] + example["sentence1"] + processed[1] + example[
            "sentence2"]

    source = tokenizer(
        text,
        max_seq_len=args.max_seq_length,
        return_token_type_ids=False,
        return_attention_mask=True,
    )

    if not args.is_test:
        return (
            source["input_ids"],
            source["attention_mask"],
            target["input_ids"],
            target["attention_mask"],
        )
    else:
        return source["input_ids"], source["attention_mask"]


def get_train_dataloader(tokenizer, args):
    filename = os.path.join("caches", args.task_name + "_train" + ".pkl")

    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("glue", args.task_name, splits="train")
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)

    batch_sampler = BatchSampler(ds,
                                 batch_size=args.train_batch_size,
                                 shuffle=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"
            ),  # attention_mask
        Pad(axis=0, pad_val=-100, dtype="int64"),  # lm_labels
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"
            ),  # decoder_attention_mask
    ): fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    return data_loader


def get_dev_dataloader(tokenizer, args):
    filename = os.path.join("caches", args.task_name + "_dev" + ".pkl")

    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("glue", args.task_name, splits="dev")
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)

    batch_sampler = BatchSampler(ds,
                                 batch_size=args.train_batch_size,
                                 shuffle=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"
            ),  # attention_mask
        Pad(axis=0, pad_val=-100, dtype="int64"),  # lm_labels
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"
            ),  # decoder_attention_mask
    ): fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    return data_loader


def get_mnli_dev_dataloader(tokenizer, args, matched=True):
    if matched:
        split = "dev_matched"
    else:
        split = "dev_mismatched"
    filename = os.path.join("caches", args.task_name + f"_{split}" + ".pkl")
    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("glue", args.task_name, splits=split)
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)

    batch_sampler = BatchSampler(ds,
                                 batch_size=args.train_batch_size,
                                 shuffle=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"
            ),  # attention_mask
        Pad(axis=0, pad_val=-100, dtype="int64"),  # lm_labels
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"
            ),  # decoder_attention_mask
    ): fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    return data_loader
