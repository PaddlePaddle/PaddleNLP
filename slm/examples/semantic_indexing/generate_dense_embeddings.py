#!/usr/bin/env python3

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

# Copyright GC-DPR authors.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import argparse
import csv
import logging
import os
import pathlib
import pickle
from typing import List, Tuple

import numpy as np
import paddle
from biencoder_base_model import BiEncoder
from NQdataset import BertTensorizer
from paddle import nn
from paddle.io import DataLoader, Dataset
from tqdm import tqdm

from paddlenlp.transformers.bert.modeling import BertModel

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


class CtxDataset(Dataset):
    def __init__(self, ctx_rows: List[Tuple[object, str, str]], tensorizer: BertTensorizer, insert_title: bool = True):
        self.rows = ctx_rows
        self.tensorizer = tensorizer
        self.insert_title = insert_title

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        ctx = self.rows[item]

        return self.tensorizer.text_to_tensor(ctx[1], title=ctx[2] if self.insert_title else None)


def no_op_collate(xx: List[object]):
    return xx


def gen_ctx_vectors(
    ctx_rows: List[Tuple[object, str, str]], model: nn.Layer, tensorizer: BertTensorizer, insert_title: bool = True
) -> List[Tuple[object, np.array]]:
    bsz = args.batch_size
    total = 0
    results = []

    dataset = CtxDataset(ctx_rows, tensorizer, insert_title)
    loader = DataLoader(
        dataset, shuffle=False, num_workers=2, collate_fn=no_op_collate, drop_last=False, batch_size=bsz
    )

    for batch_id, batch_token_tensors in enumerate(tqdm(loader)):
        ctx_ids_batch = paddle.stack(batch_token_tensors, axis=0)
        ctx_seg_batch = paddle.zeros_like(ctx_ids_batch)
        with paddle.no_grad():
            out = model.get_context_pooled_embedding(ctx_ids_batch, ctx_seg_batch)

        out = out.astype("float32").cpu()
        batch_start = batch_id * bsz
        ctx_ids = [r[0] for r in ctx_rows[batch_start : batch_start + bsz]]
        assert len(ctx_ids) == out.shape[0]
        total += len(ctx_ids)
        results.extend([(ctx_ids[i], out[i].reshape([-1]).numpy()) for i in range(out.shape[0])])

    return results


def main(args):

    tensorizer = BertTensorizer()
    question_model = BertModel.from_pretrained(args.que_model_path)
    context_model = BertModel.from_pretrained(args.con_model_path)
    model = BiEncoder(question_encoder=question_model, context_encoder=context_model)

    rows = []
    with open(args.ctx_file) as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t")
        # file format: doc_id, doc_text, title
        rows.extend([(row[0], row[1], row[2]) for row in reader if row[0] != "id"])

    shard_size = int(len(rows) / args.num_shards)
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size

    logger.info("Producing encodings for passages range: %d to %d (out of total %d)", start_idx, end_idx, len(rows))
    rows = rows[start_idx:end_idx]
    data = gen_ctx_vectors(rows, model, tensorizer, True)
    file = args.out_file + "_" + str(args.shard_id) + ".pkl"
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info("Writing results to %s" % file)
    with open(file, mode="wb") as f:
        pickle.dump(data, f)

    logger.info("Total passages processed %d. Written to %s", len(data), file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ctx_file", type=str, default=None, help="Path to passages set .tsv file")
    parser.add_argument(
        "--out_file", required=True, type=str, default=None, help="output file path to write results to"
    )
    parser.add_argument("--shard_id", type=int, default=0, help="Number(0-based) of data shard to process")
    parser.add_argument("--num_shards", type=int, default=1, help="Total amount of data shards")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the passage encoder forward pass")
    parser.add_argument("--que_model_path", type=str)
    parser.add_argument("--con_model_path", type=str)
    args = parser.parse_args()

    main(args)
