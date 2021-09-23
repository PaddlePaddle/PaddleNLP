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
from functools import partial
import argparse
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F

import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad, Vocab
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers import ErnieModel, ErnieTokenizer
from paddlenlp.utils.log import logger

from model import ErnieForCSC
from utils import read_test_ds, convert_example, create_dataloader, is_chinese_char, parse_decode

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="ernie-1.0", choices=["ernie-1.0"], help="Pretraining model name or path")
parser.add_argument("--ckpt_path", default=None, type=str, help="The model checkpoint path.", )
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer " "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu"] ,help="The device to select to train the model, is must be cpu/gpu/xpu.")
parser.add_argument("--pinyin_vocab_file_path", type=str, default="pinyin_vocab.txt", help="pinyin vocab file path")
parser.add_argument("--test_file", type=str, default="test.txt", help="test set file")
parser.add_argument("--predict_file", type=str, default="predict.txt", help="predict result file")

# yapf: enable
args = parser.parse_args()


def write_sighan_result_to_file(args, corr_preds, det_preds, lengths,
                                tokenizer):
    with open(args.test_file, 'r', encoding='utf-8') as fin:
        with open(args.predict_file, 'w', encoding='utf-8') as fout:
            for i, line in enumerate(fin.readlines()):
                ids, words = line.strip('\n').split('\t')[0:2]
                ids = ids.split('=')[1][:-1]
                pred_result = parse_decode(words, corr_preds[i], det_preds[i],
                                           lengths[i], tokenizer,
                                           args.max_seq_length)
                words = list(words)
                pred_result = list(pred_result)
                result = ids
                if pred_result == words:
                    result += ', 0'
                else:
                    assert len(pred_result) == len(
                        words), "pred_result: {}, words: {}".format(pred_result,
                                                                    words)
                    for i, word in enumerate(pred_result):
                        if word != words[i]:
                            result += ', {}, {}'.format(i + 1, word)
                fout.write("{}\n".format(result))


@paddle.no_grad()
def do_predict(args):
    paddle.set_device(args.device)

    pinyin_vocab = Vocab.load_vocabulary(
        args.pinyin_vocab_file_path, unk_token='[UNK]', pad_token='[PAD]')

    tokenizer = ErnieTokenizer.from_pretrained(args.model_name_or_path)
    ernie = ErnieModel.from_pretrained(args.model_name_or_path)

    model = ErnieForCSC(
        ernie,
        pinyin_vocab_size=len(pinyin_vocab),
        pad_pinyin_id=pinyin_vocab[pinyin_vocab.pad_token])

    eval_ds = load_dataset(read_test_ds, data_path=args.test_file, lazy=False)
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        pinyin_vocab=pinyin_vocab,
        max_seq_length=args.max_seq_length,
        is_test=True)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),  # segment
        Pad(axis=0, pad_val=pinyin_vocab.token_to_idx[pinyin_vocab.pad_token], dtype='int64'),  # pinyin
        Stack(axis=0, dtype='int64'),  # length
    ): [data for data in fn(samples)]

    test_data_loader = create_dataloader(
        eval_ds,
        mode='test',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    if args.ckpt_path:
        model_dict = paddle.load(args.ckpt_path)
        model.set_dict(model_dict)
        logger.info("Load model from checkpoints: {}".format(args.ckpt_path))

    model.eval()
    corr_preds = []
    det_preds = []
    lengths = []
    for step, batch in enumerate(test_data_loader):
        input_ids, token_type_ids, pinyin_ids, length = batch
        det_error_probs, corr_logits = model(input_ids, pinyin_ids,
                                             token_type_ids)
        # corr_logits shape: [B, T, V]
        det_pred = det_error_probs.argmax(axis=-1)
        det_pred = det_pred.numpy()

        char_preds = corr_logits.argmax(axis=-1)
        char_preds = char_preds.numpy()

        length = length.numpy()

        corr_preds += [pred for pred in char_preds]
        det_preds += [prob for prob in det_pred]
        lengths += [l for l in length]

    write_sighan_result_to_file(args, corr_preds, det_preds, lengths, tokenizer)


if __name__ == "__main__":
    do_predict(args)
