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


def parse_args():
    parser = argparse.ArgumentParser()

    # yapf: disable
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir, should contain [train/test].json and [train/test]_metrics.json")
    parser.add_argument("--init_ckpt_dir", default="ernie-ctm", type=str, help="The pre-trained model checkpoint dir.")
    parser.add_argument("--max_seq_len", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.", )
    parser.add_argument("--device", default="gpu", type=str, help="The device to select to train the model, is must be cpu/gpu/xpu.")
    # yapf: enable

    args = parser.parse_args()
    return args


def do_predict(args):
    paddle.set_device(args.device)
    predictor = WordtagPredictor(args.init_ckpt_dir, "./data/tags.txt")
    txts = ['《孤女》是2010年九州出版社出版的小说，作者是余兼羽。', '4分40秒至10分钟只有歌声。']
    res = predictor.run(txts)
    print(res)


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    do_predict(args)
