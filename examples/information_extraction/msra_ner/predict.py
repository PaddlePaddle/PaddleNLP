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
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.transformers import BertForTokenClassification, BertTokenizer

parser = argparse.ArgumentParser()

# yapf: disable
parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(list(BertTokenizer.pretrained_init_configuration.keys())))
parser.add_argument("--init_checkpoint_path", default=None, type=str, required=True, help="The model checkpoint path.", )
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer " "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu", "xpu"] ,help="The device to select to train the model, is must be cpu/gpu/xpu.")
# yapf: enable


def tokenize_and_align_labels(example, tokenizer, no_entity_id,
                              max_seq_len=512):
    labels = example['labels']
    example = example['tokens']
    tokenized_input = tokenizer(
        example,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)

    # -2 for [CLS] and [SEP]
    if len(tokenized_input['input_ids']) - 2 < len(labels):
        labels = labels[:len(tokenized_input['input_ids']) - 2]
    tokenized_input['labels'] = [no_entity_id] + labels + [no_entity_id]
    tokenized_input['labels'] += [no_entity_id] * (
        len(tokenized_input['input_ids']) - len(tokenized_input['labels']))

    return tokenized_input


def parse_decodes(input_words, id2label, decodes, lens):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]

    outputs = []
    for idx, end in enumerate(lens):
        sent = "".join(input_words[idx]['tokens'])
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
    paddle.set_device(args.device)

    # Create dataset, tokenizer and dataloader.
    train_ds, predict_ds = load_dataset(
        'msra_ner', splits=('train', 'test'), lazy=False)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    label_list = train_ds.label_list
    label_num = len(label_list)
    no_entity_id = label_num - 1
    trans_func = partial(
        tokenize_and_align_labels,
        tokenizer=tokenizer,
        no_entity_id=no_entity_id,
        max_seq_len=args.max_seq_length)

    ignore_label = -100
    batchify_fn = lambda samples, fn=Dict({
        'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        'seq_len': Stack(),
        'labels': Pad(axis=0, pad_val=ignore_label)  # label
    }): fn(samples)
    raw_data = predict_ds.data

    id2label = dict(enumerate(predict_ds.label_list))

    predict_ds = predict_ds.map(trans_func)
    predict_data_loader = DataLoader(
        dataset=predict_ds,
        collate_fn=batchify_fn,
        num_workers=0,
        batch_size=args.batch_size,
        return_list=True)

    # Define the model netword
    model = BertForTokenClassification.from_pretrained(
        args.model_name_or_path, num_classes=label_num)
    if args.init_checkpoint_path:
        model_dict = paddle.load(args.init_checkpoint_path)
        model.set_dict(model_dict)

    model.eval()
    pred_list = []
    len_list = []
    for step, batch in enumerate(predict_data_loader):
        input_ids, token_type_ids, length, labels = batch
        logits = model(input_ids, token_type_ids)
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
