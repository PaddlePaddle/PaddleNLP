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

import argparse
import os
import ast
import random
import time
import math
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader

import paddlenlp as ppnlp
from paddlenlp.datasets import MSRA_NER
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import BertForTokenClassification, BertTokenizer

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
    help="Path to pre-trained model or shortcut name selected in the list: " +
    ", ".join(list(BertTokenizer.pretrained_init_configuration.keys())))
parser.add_argument(
    "--init_checkpoint_path",
    default=None,
    type=str,
    required=True,
    help="The model checkpoint path.", )
parser.add_argument(
    "--max_seq_length",
    default=128,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument(
    "--batch_size",
    default=8,
    type=int,
    help="Batch size per GPU/CPU for training.", )
parser.add_argument(
    "--use_gpu",
    type=ast.literal_eval,
    default=True,
    help="If set, use GPU for training.")


def convert_example(example,
                    tokenizer,
                    label_list,
                    no_entity_id,
                    max_seq_length=512,
                    is_test=False):
    """convert a glue example into necessary features"""

    def _truncate_seqs(seqs, max_seq_length):
        if len(seqs) == 1:  # single sentence
            # Account for [CLS] and [SEP] with "- 2"
            seqs[0] = seqs[0][0:(max_seq_length - 2)]
        else:  # sentence pair
            # Account for [CLS], [SEP], [SEP] with "- 3"
            tokens_a, tokens_b = seqs
            max_seq_length -= 3
            while True:  # truncate with longest_first strategy
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_seq_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()
        return seqs

    def _concat_seqs(seqs, separators, seq_mask=0, separator_mask=1):
        concat = sum((seq + sep for sep, seq in zip(separators, seqs)), [])
        segment_ids = sum(
            ([i] * (len(seq) + len(sep))
             for i, (sep, seq) in enumerate(zip(separators, seqs))), [])
        if isinstance(seq_mask, int):
            seq_mask = [[seq_mask] * len(seq) for seq in seqs]
        if isinstance(separator_mask, int):
            separator_mask = [[separator_mask] * len(sep) for sep in separators]
        p_mask = sum((s_mask + mask
                      for sep, seq, s_mask, mask in zip(
                          separators, seqs, seq_mask, separator_mask)), [])
        return concat, segment_ids, p_mask

    def _reseg_token_label(tokens, tokenizer, labels=None):
        if labels:
            if len(tokens) != len(labels):
                raise ValueError(
                    "The length of tokens must be same with labels")
            ret_tokens = []
            ret_labels = []
            for token, label in zip(tokens, labels):
                sub_token = tokenizer(token)
                if len(sub_token) == 0:
                    continue
                ret_tokens.extend(sub_token)
                ret_labels.append(label)
                if len(sub_token) < 2:
                    continue
                sub_label = label
                if label.startswith("B-"):
                    sub_label = "I-" + label[2:]
                ret_labels.extend([sub_label] * (len(sub_token) - 1))

            if len(ret_tokens) != len(ret_labels):
                raise ValueError(
                    "The length of ret_tokens can't match with labels")
            return ret_tokens, ret_labels
        else:
            ret_tokens = []
            for token in tokens:
                sub_token = tokenizer(token)
                if len(sub_token) == 0:
                    continue
                ret_tokens.extend(sub_token)
                if len(sub_token) < 2:
                    continue

            return ret_tokens, None

    if not is_test:
        # get the label
        label = example[-1].split("\002")
        example = example[0].split("\002")
        #create label maps if classification task
        label_map = {}
        for (i, l) in enumerate(label_list):
            label_map[l] = i
    else:
        label = None

    tokens_raw, labels_raw = _reseg_token_label(
        tokens=example, labels=label, tokenizer=tokenizer)
    # truncate to the truncate_length,
    tokens_trun = _truncate_seqs([tokens_raw], max_seq_length)
    # concate the sequences with special tokens
    tokens_trun[0] = [tokenizer.cls_token] + tokens_trun[0]
    tokens, segment_ids, _ = _concat_seqs(tokens_trun, [[tokenizer.sep_token]] *
                                          len(tokens_trun))
    # convert the token to ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    valid_length = len(input_ids)
    if labels_raw:
        labels_trun = _truncate_seqs([labels_raw], max_seq_length)[0]
        labels_id = [no_entity_id] + [label_map[lbl]
                                      for lbl in labels_trun] + [no_entity_id]
    if not is_test:
        return input_ids, segment_ids, valid_length, labels_id
    else:
        return input_ids, segment_ids, valid_length


def parse_decodes(input_words, id2label, decodes, lens):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]

    outputs = []
    for idx, end in enumerate(lens):
        sent = input_words[idx][0].replace("\002", "")[:end]
        tags = [id2label[x] for x in decodes[idx][1:end]]
        sent_out = []
        tags_out = []
        words = ""
        for s, t in zip(sent, tags):
            if t.startswith('B-') or t == 'O':
                if len(words):
                    sent_out.append(words)
                if t.startswith('B-'):
                    tags_out.append(t.split('-')[1])
                else:
                    tags_out.append(t)
                words = s
            else:
                words += s
        if len(sent_out) < len(tags_out):
            sent_out.append(words)
        outputs.append(''.join(
            [str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return outputs


def do_predict(args):
    paddle.set_device("gpu" if args.use_gpu else "cpu")

    train_dataset, predict_dataset = ppnlp.datasets.MSRA_NER.get_datasets(
        ["train", "test"])
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    label_list = train_dataset.get_labels()
    label_num = len(label_list)
    no_entity_id = label_num - 1
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=label_list,
        no_entity_id=label_num - 1,
        max_seq_length=args.max_seq_length)
    ignore_label = -100
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # input
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # segment
        Stack(),  # length
        Pad(axis=0, pad_val=ignore_label)  # label
    ): fn(samples)
    raw_data = predict_dataset.data

    id2label = dict(enumerate(predict_dataset.get_labels()))

    predict_dataset = predict_dataset.apply(trans_func, lazy=True)
    predict_batch_sampler = paddle.io.BatchSampler(
        predict_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True)
    predict_data_loader = DataLoader(
        dataset=predict_dataset,
        batch_sampler=predict_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    model = BertForTokenClassification.from_pretrained(
        args.model_name_or_path, num_classes=label_num)
    if args.init_checkpoint_path:
        model_dict = paddle.load(args.init_checkpoint_path)
        model.set_dict(model_dict)

    model.eval()
    pred_list = []
    len_list = []
    for step, batch in enumerate(predict_data_loader):
        input_ids, segment_ids, length, labels = batch
        logits = model(input_ids, segment_ids)
        pred = paddle.argmax(logits, axis=-1)
        pred_list.append(pred.numpy())
        len_list.append(length.numpy())

    preds = parse_decodes(raw_data, id2label, pred_list, len_list)

    file_path = "results.txt"
    with open(file_path, "w", encoding="utf8") as fout:
        fout.write("\n".join(preds))
    # Print some examples
    print(
        "The results have been saved in the file: %s, some examples are shown below: "
        % file_path)
    print("\n".join(preds[:10]))


if __name__ == "__main__":
    args = parser.parse_args()
    do_predict(args)
