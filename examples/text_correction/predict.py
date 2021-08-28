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
from paddlenlp.transformers import ErnieGramTokenizer
from paddlenlp.utils.log import logger

from model import ErnieGramForCSC
from data import read_test_ds, convert_example, create_dataloader, is_chinese_char

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", default='ernie-gram-zh', type=str, help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(list(ErnieGramTokenizer.pretrained_init_configuration.keys())))
parser.add_argument("--init_checkpoint_path", default=None, type=str, help="The model checkpoint path.", )
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer " "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu"] ,help="The device to select to train the model, is must be cpu/gpu/xpu.")
parser.add_argument("--pinyin_vocab_file_path", type=str, default="pinyin_vocab.txt", help="pinyin vocab file path")
parser.add_argument("--test_file", type=str, default="test.txt", help="test set file")
parser.add_argument("--predict_file", type=str, default="predict.txt", help="predict result file")

# yapf: enable
args = parser.parse_args()


def write_result_to_file(args, corr_preds, det_preds, lengths, tokenizer):
    with open(args.test_file, 'r', encoding='utf-8') as fin:
        with open(args.predict_file, 'w', encoding='utf-8') as fout:
            for i, line in enumerate(fin.readlines()):
                UNK = tokenizer.unk_token
                UNK_id = tokenizer.convert_tokens_to_ids(UNK)

                ids, words = line.strip('\n').split('\t')[0:2]
                ids = ids.split('=')[1][:-1]
                tokens = tokenizer.tokenize(words)
                if len(tokens) > args.max_seq_length - 2:
                    tokens = tokens[:args.max_seq_length - 2]
                corr_pred = corr_preds[i][1:1 + lengths[i]].tolist()
                det_pred = det_preds[i][1:1 + lengths[i]].tolist()

                words = list(words)
                if len(words) > args.max_seq_length - 2:
                    words = words[:args.max_seq_length - 2]

                assert len(tokens) == len(
                    corr_pred
                ), "The number of tokens should be equal to the number of labels {}: {}: {}".format(
                    len(tokens), len(corr_pred), tokens)
                pred_result = ""

                # need to be aligned
                align_offset = 0
                if len(words) != len(tokens):
                    first_unk_flag = True
                    for j, word in enumerate(words):
                        if word.isspace():
                            tokens.insert(j + 1, word)
                            corr_pred.insert(j + 1, UNK_id)
                            det_pred.insert(j + 1, 0)  # no error
                        elif tokens[j] != word:
                            if tokenizer.convert_tokens_to_ids(word) == UNK_id:
                                if first_unk_flag:
                                    first_unk_flag = False
                                    corr_pred[j] = UNK_id
                                    det_pred[j] = 0
                                else:
                                    tokens.insert(j, UNK)
                                    corr_pred.insert(j, UNK_id)
                                    det_pred.insert(j, 0)  # no error
                                continue
                            elif tokens[j] == UNK:
                                # remove rest unk
                                k = 0
                                while k + j < len(tokens) and tokens[k +
                                                                     j] == UNK:
                                    k += 1
                                tokens = tokens[:j] + tokens[j + k:]
                                corr_pred = corr_pred[:j] + corr_pred[j + k:]
                                det_pred = det_pred[:j] + det_pred[j + k:]
                            else:  # maybe English, number
                                if tokens[j].isalnum():
                                    corr_pred = corr_pred[:j] + [UNK_id] * len(
                                        tokens[j]) + corr_pred[j + 1:]
                                    det_pred = det_pred[:j] + [0] * len(tokens[
                                        j]) + det_pred[j + 1:]
                                    tokens = tokens[:j] + list(tokens[
                                        j]) + tokens[j + 1:]
                        first_unk_flag = True

                # print("tokens:", tokens)
                # print("words: ", words)
                for j, word in enumerate(words):
                    candidates = tokenizer.convert_ids_to_tokens(corr_pred[j])
                    if det_pred[
                            j] == 0 or candidates == UNK or candidates == '[PAD]':
                        pred_result += word
                    else:
                        pred_result += candidates

                result = ids
                words = ''.join(words)
                if pred_result == words:
                    result += ', 0'
                else:
                    pred_result = list(pred_result)
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
    tokenizer = ErnieGramTokenizer.from_pretrained(args.model_name_or_path)

    model = ErnieGramForCSC.from_pretrained(
        args.model_name_or_path,
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
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Pad(axis=0, pad_val=pinyin_vocab.token_to_idx[pinyin_vocab.pad_token]),  # pinyin
        Stack(axis=0),  # length
    ): [data for data in fn(samples)]

    test_data_loader = create_dataloader(
        eval_ds,
        mode='test',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    if args.init_checkpoint_path:
        model_dict = paddle.load(args.init_checkpoint_path)
        model.set_dict(model_dict)
        logger.info("Load model from checkpoints: {}".format(
            args.init_checkpoint_path))

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

    write_result_to_file(args, corr_preds, det_preds, lengths, tokenizer)


if __name__ == "__main__":
    do_predict(args)
