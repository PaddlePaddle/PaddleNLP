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

from datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.transformers import BertForTokenClassification, BertTokenizer
from paddlenlp.metrics import ChunkEvaluator

parser = argparse.ArgumentParser()

# yapf: disable
parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(list(BertTokenizer.pretrained_init_configuration.keys())))
parser.add_argument("--init_checkpoint_path", default=None, type=str, required=True, help="The model checkpoint path.", )
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer " "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu", "xpu"] ,help="The device to select to train the model, is must be cpu/gpu/xpu.")
# yapf: enable


def do_eval(args):
    paddle.set_device(args.device)

    # Create dataset, tokenizer and dataloader.
    train_ds, eval_ds = load_dataset('msra_ner', split=('train', 'test'))
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    label_list = train_ds.features['ner_tags'].feature.names
    label_num = len(label_list)
    no_entity_id = 0

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples['tokens'],
            max_seq_len=args.max_seq_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
            return_length=True)
        labels = []

        for i, label in enumerate(examples['ner_tags']):
            label_ids = label
            if len(tokenized_inputs['input_ids'][i]) - 2 < len(label_ids):
                label_ids = label_ids[:len(tokenized_inputs['input_ids'][i]) -
                                      2]
            label_ids = [no_entity_id] + label_ids + [no_entity_id]
            label_ids += [no_entity_id] * (
                len(tokenized_inputs['input_ids'][i]) - len(label_ids))

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    ignore_label = -100
    batchify_fn = lambda samples, fn=Dict({
        'input_ids':
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int32'),  # input
        'token_type_ids':
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int32'
            ),  # segment
        'seq_len':
        Stack(dtype='int64'),
        'labels':
        Pad(axis=0, pad_val=ignore_label, dtype='int64')  # label
    }): fn(samples)

    eval_ds = eval_ds.select(range(len(eval_ds) - 1))
    eval_ds = eval_ds.map(tokenize_and_align_labels, batched=True)
    eval_data_loader = DataLoader(dataset=eval_ds,
                                  collate_fn=batchify_fn,
                                  num_workers=0,
                                  batch_size=args.batch_size,
                                  return_list=True)

    # Define the model netword and its loss
    model = BertForTokenClassification.from_pretrained(args.model_name_or_path,
                                                       num_classes=label_num)
    if args.init_checkpoint_path:
        model_dict = paddle.load(args.init_checkpoint_path)
        model.set_dict(model_dict)
    loss_fct = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)

    metric = ChunkEvaluator(label_list=label_list)

    model.eval()
    metric.reset()
    for step, batch in enumerate(eval_data_loader):
        input_ids, token_type_ids, length, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = loss_fct(logits, labels)
        avg_loss = paddle.mean(loss)
        preds = logits.argmax(axis=2)
        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(
            length, preds, labels)
        metric.update(num_infer_chunks.numpy(), num_label_chunks.numpy(),
                      num_correct_chunks.numpy())
        precision, recall, f1_score = metric.accumulate()
    print("eval loss: %f, precision: %f, recall: %f, f1: %f" %
          (avg_loss, precision, recall, f1_score))


if __name__ == "__main__":
    args = parser.parse_args()
    do_eval(args)
