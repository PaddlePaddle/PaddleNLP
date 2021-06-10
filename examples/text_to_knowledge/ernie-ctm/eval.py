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

import io
import sys
import argparse

import paddle

from predictor import WordtagPredictor
from metric import wordseg_hard_acc, wordtag_hard_acc


def parse_args():
    parser = argparse.ArgumentParser()

    # yapf: disable
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir, should contain [train/test].json and [train/test]_metrics.json")
    parser.add_argument("--init_ckpt_dir", default="ernie-ctm", type=str, help="The pre-trained model checkpoint dir.")
    parser.add_argument("--max_seq_len", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.", )
    parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu", "xpu"] ,help="The device to select to train the model, is must be cpu/gpu/xpu.")
    # yapf: enable

    args = parser.parse_args()
    return args


def do_eval(args):
    paddle.set_device(args.device)
    predictor = WordtagPredictor(args.init_ckpt_dir, tag_path="./data/tags.txt")
    total_real = 0.0
    total_pred = 0.0
    seg_acc = 0.0
    tag_acc = 0.0
    for line in open("./data/eval.txt", encoding='utf8'):
        line = line.strip()
        wl = line.split("\t")
        text = wl[0]
        res = predictor.run(text)
        pred_words = [(r["item"], r["wordtag_label"]) for r in res[0]["items"]]
        real_words = [item.split("\\") for item in wl[1].split("  ")]

        seg_acc += wordseg_hard_acc(pred_words, real_words)
        tag_acc += wordtag_hard_acc(pred_words, real_words)
        total_pred += float(len(pred_words))
        total_real += float(len(real_words))
    precision = seg_acc / total_pred
    recall = seg_acc / total_real
    f1 = 2.0 * precision * recall / (precision + recall)

    print(f"Precision: {precision:g}, Recall: {recall:g}, F1: {f1:g}")

    precision = tag_acc / total_pred
    recall = tag_acc / total_real
    f1 = 2.0 * precision * recall / (precision + recall)

    print(f"Precision: {precision:g}, Recall: {recall:g}, F1: {f1:g}")


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    do_eval(args)
