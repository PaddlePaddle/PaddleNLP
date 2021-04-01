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
import argparse
import time
import random
from functools import partial

import paddle
from paddle.io import DataLoader
import numpy as np
from paddlenlp.utils.log import logger
from paddlenlp.transformers import ErnieCtmWordtagModel, ErnieCtmTokenizer, LinearDecayWithWarmup
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.data import Stack, Pad, Tuple

from data import load_dataset, load_dict, convert_example
from metric import SequenceAccuracy


def parse_args():
    parser = argparse.ArgumentParser()

    # yapf: disable
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir, should contain [train/test].json and [train/test]_metrics.json")
    parser.add_argument("--model_dir", default="ernie-ctm", type=str, help="The pre-trained model checkpoint dir.")
    parser.add_argument("--max_seq_len",default=128,type=int,help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--batch_size",default=32,type=int,help="Batch size per GPU/CPU for training.", )
    parser.add_argument("--n_gpu",default=1,type=int,help="number of gpus to use, 0 for cpu.")
    # yapf: enable

    args = parser.parse_args()
    return args


def do_predict(args):
    paddle.set_device("gpu" if args.n_gpu else "cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    raw_test_ds = load_dataset(datafiles=('./data/test.json'))
    test_ds = load_dataset(datafiles=('./data/test.json'))
    tags_to_idx = load_dict("./data/tags.txt")
    idx_to_tags = {v: k for k, v in tags_to_idx.items()}
    labels_to_idx = load_dict("./data/classifier_labels.txt")
    idx_to_labels = {v: k for k, v in labels_to_idx.items()}
    tokenizer = ErnieCtmTokenizer.from_pretrained(args.model_dir)
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        tags_to_idx=tags_to_idx,
        labels_to_idx=labels_to_idx)
    test_ds.map(trans_func)

    ignore_label = tags_to_idx["O"]
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
        Stack(),  # seq_len
        Pad(axis=0, pad_val=ignore_label),  # tags
        Stack(),  # cls_label
    ): fn(samples)

    test_data_loader = DataLoader(
        test_ds,
        collate_fn=batchify_fn,
        num_workers=0,
        batch_size=args.batch_size,
        shuffle=False,
        return_list=True)

    model = ErnieCtmWordtagModel.from_pretrained(
        args.model_dir,
        num_cls_label=len(labels_to_idx),
        num_tag=len(tags_to_idx),
        ignore_index=tags_to_idx["O"])

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    model.eval()
    for batch in test_data_loader:
        input_ids, token_type_ids, seq_len, tags, cls_label = batch
        seq_logits, cls_logits = model(
            input_ids, token_type_ids, lengths=seq_len)
        scores, pred_tags = model.viterbi_decoder(seq_logits, seq_len)

        pred_cls_label = paddle.argmax(cls_logits, axis=-1, keepdim=False)
        seq_len_np = seq_len.numpy().tolist()
        for i, pred_tag in enumerate(pred_tags):
            pred_tag = pred_tag[:seq_len_np[i]]
            sent = tokenizer.convert_ids_to_tokens(input_ids[i][:seq_len_np[i]]
                                                   .numpy().tolist())
            # Drop CLS and SEP
            sent = sent[2:-1]
            if pred_cls_label[i] != labels_to_idx["其他文本"]:
                print("%s is %s, which is not supported." %
                      (''.join(sent), idx_to_labels[pred_cls_label[i]]))
            tags = list(map(idx_to_tags.get, pred_tag.numpy()))
            # Drop CLS and SEP
            tags = tags[2:-1]
            # import pdb; pdb.set_trace()

            # Parse the result
            sent_out = []
            tags_out = []
            parital_word = ""
            for ind, tag in enumerate(tags):
                # for the first word
                if parital_word == "":
                    parital_word = sent[ind]
                    if tag == "O":
                        tags_out.append(tag)
                    else:
                        tags_out.append(tag.split('-')[1])
                    continue

                # for the beginning of word
                if tag.startswith("B-") or (
                        tag == "O" and
                        tags[ind - 1] != "O") or tag.startswith("S-"):
                    sent_out.append(parital_word)
                    if tag == "O":
                        tags_out.append(tag)
                    else:
                        tags_out.append(tag.split('-')[1])
                    parital_word = sent[ind]
                    continue

                parital_word += sent[ind]

            # append the last word, except for len(tags)=0
            if len(sent_out) < len(tags_out):
                sent_out.append(parital_word)
            sent_tag = [
                '(%s, %s)' % (ch, tag) for ch, tag in zip(sent_out, tags_out)
            ]
            print("".join(sent_tag))


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    if args.n_gpu > 1:
        paddle.distributed.spawn(do_predict, args=(args, ), nprocs=args.n_gpu)
    else:
        do_predict(args)
