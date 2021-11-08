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
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
from paddlenlp.experimental import FasterErnieForTokenClassification, to_tensor

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default="ckpt/model_4000", help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", type=int, default=128, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def batchify_fn(batch, no_entity_id, ignore_label=-100, max_seq_len=512):
    texts, labels, seq_lens = [], [], []
    for example in batch:
        texts.append("".join(example["tokens"]))
        # 2 for [CLS] and [SEP]
        seq_lens.append(len(example["tokens"]) + 2)
        label = example["labels"]
        if len(label) > max_seq_len - 2:
            label = label[:(max_seq_len - 2)]
        label = [no_entity_id] + label + [no_entity_id]
        if len(label) < max_seq_len:
            label += [ignore_label] * (max_seq_len - len(label))
        labels.append(label)

    labels = np.array(labels)
    seq_lens = np.array(seq_lens)
    return texts, labels, seq_lens


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


def do_predict():
    paddle.set_device(args.device)

    # Create dataset, tokenizer and dataloader.
    predict_ds = load_dataset('msra_ner', splits=('test'), lazy=False)
    model = FasterErnieForTokenClassification.from_pretrained(
        args.save_path,
        num_classes=len(predict_ds.label_list),
        max_seq_len=args.max_seq_length,
        is_split_into_words=True)

    # ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'O']
    label_num = len(predict_ds.label_list)
    # the label 'O'  index
    no_entity_id = label_num - 1
    # ignore_label is for the label padding
    ignore_label = -100
    trans_func = partial(
        batchify_fn,
        no_entity_id=no_entity_id,
        ignore_label=ignore_label,
        max_seq_len=args.max_seq_length)
    data_loader = paddle.io.DataLoader(
        dataset=predict_ds,
        batch_size=args.batch_size,
        collate_fn=trans_func,
        return_list=True)

    model.eval()
    pred_list = []
    len_list = []
    for texts, labels, seq_lens in data_loader:
        texts = to_tensor(texts)
        logits, preds = model(texts)
        pred_list.append(preds.numpy())
        len_list.append(seq_lens.numpy())

    raw_data = predict_ds.data
    id2label = dict(enumerate(predict_ds.label_list))
    print(id2label)
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
    do_predict()
