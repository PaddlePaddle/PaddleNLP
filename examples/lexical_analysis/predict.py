# -*- coding: UTF-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import ast
import argparse

import numpy as np
import paddle
from paddlenlp.data import Pad, Tuple, Stack
from paddlenlp.metrics import ChunkEvaluator

from data import LacDataset, parse_lac_result
from model import BiGruCrf

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--data_dir", type=str, default=None, help="The folder where the dataset is located.")
parser.add_argument("--init_checkpoint", type=str, default=None, help="Path to init model.")
parser.add_argument("--batch_size", type=int, default=300, help="The number of sequences contained in a mini-batch.")
parser.add_argument("--max_seq_len", type=int, default=64, help="Number of words of the longest seqence.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="If set, use GPU for training.")
parser.add_argument("--emb_dim", type=int, default=128, help="The dimension in which a word is embedded.")
parser.add_argument("--hidden_size", type=int, default=128, help="The number of hidden nodes in the GRU layer.")
args = parser.parse_args()
# yapf: enable


def infer(args):
    place = paddle.CUDAPlace(0) if args.use_gpu else paddle.CPUPlace()
    paddle.set_device("gpu" if args.use_gpu else "cpu")

    # create dataset.
    infer_dataset = LacDataset(args.data_dir, mode='infer')

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=0),  # word_ids
        Stack(),  # length
    ): fn(samples)

    # Create sampler for dataloader
    infer_sampler = paddle.io.BatchSampler(
        dataset=infer_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False)
    infer_loader = paddle.io.DataLoader(
        dataset=infer_dataset,
        batch_sampler=infer_sampler,
        places=place,
        return_list=True,
        collate_fn=batchify_fn)

    # Define the model network
    network = BiGruCrf(args.emb_dim, args.hidden_size, infer_dataset.vocab_size,
                       infer_dataset.num_labels)
    model = paddle.Model(network)
    model.prepare()

    # Load the model and start predicting
    model.load(args.init_checkpoint)
    emissions, lengths, crf_decodes = model.predict(
        test_data=infer_loader, batch_size=args.batch_size)

    # Post-processing the lexical analysis results
    lengths = np.array([l for lens in lengths for l in lens]).reshape([-1])
    preds = np.array(
        [pred for batch_pred in crf_decodes for pred in batch_pred])

    results = parse_lac_result(infer_dataset.word_ids, preds, lengths,
                               infer_dataset.word_vocab,
                               infer_dataset.label_vocab)

    sent_tags = []
    for sent, tags in results:
        sent_tag = ['(%s, %s)' % (ch, tag) for ch, tag in zip(sent, tags)]
        sent_tags.append(''.join(sent_tag))

    file_path = "results.txt"
    with open(file_path, "w", encoding="utf8") as fout:
        fout.write("\n".join(sent_tags))

    # Print some examples
    print(
        "The results have been saved in the file: %s, some examples are shown below: "
        % file_path)
    print("\n".join(sent_tags[:10]))


if __name__ == '__main__':
    infer(args)
