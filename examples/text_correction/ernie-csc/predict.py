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

from model import PretrainedModelForCSC
from utils import read_test_ds, convert_example, is_chinese_char, parse_decode

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="ernie-1.0", choices=["ernie-1.0"], help="Pretraining model name or path")
parser.add_argument("--init_checkpoint_path", default=None, type=str, help="The model checkpoint path.", )
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer " "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu"] ,help="The device to select to train the model, is must be cpu/gpu/xpu.")
parser.add_argument("--pinyin_vocab_file_path", type=str, default="pinyin_vocab.txt", help="pinyin vocab file path")

# yapf: enable
args = parser.parse_args()

MODEL_CLASSES = {
    "ernie_gram": (ErnieGramModel, ErnieGramTokenizer),
    "ernie": (ErnieModel, ErnieTokenizer),
    "roberta": (RobertaModel, RobertaTokenizer)
}


@paddle.no_grad()
def do_predict(args):
    paddle.set_device(args.device)

    pinyin_vocab = Vocab.load_vocabulary(
        args.pinyin_vocab_file_path, unk_token='[UNK]', pad_token='[PAD]')

    MODEL_CLASS, TOKENIZER_CLASS = MODEL_CLASSES[args.model_type]
    tokenizer = TOKENIZER_CLASS.from_pretrained(args.model_name_or_path)
    pretrained_model = MODEL_CLASS.from_pretrained(args.model_name_or_path)

    model = PretrainedModelForCSC(
        pretrained_model,
        pinyin_vocab_size=len(pinyin_vocab),
        pad_pinyin_id=pinyin_vocab[pinyin_vocab.pad_token])

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
        Stack(axis=0, dtype='int64'),  # length
    ): [data for data in fn(samples)]

    if args.init_checkpoint_path:
        model_dict = paddle.load(args.init_checkpoint_path)
        model.set_dict(model_dict)
        logger.info("Load model from checkpoints: {}".format(
            args.init_checkpoint_path))

    while True:
        try:
            source = input("请输入待纠错样本：")
            if source.lower() == "exit":
                break
            example = {"source": source}
            input_ids, token_type_ids, pinyin_ids, length = trans_func(example)
            input_ids, token_type_ids, pinyin_ids, length = map(
                lambda x: paddle.to_tensor(x),
                batchify_fn([[input_ids, token_type_ids, pinyin_ids, length]]))
            det_error_probs, corr_logits = model(input_ids, pinyin_ids,
                                                 token_type_ids)

            det_pred = det_error_probs.argmax(axis=-1)
            det_pred = det_pred.numpy()

            char_preds = corr_logits.argmax(axis=-1)
            char_preds = char_preds.numpy()

            length = length.numpy()

            pred_result = parse_decode(source, char_preds[0], det_pred[0],
                                       length[0], tokenizer,
                                       args.max_seq_length)
            print("纠正后结果：", pred_result)
        except EOFError:
            print("Exit.")
            break


if __name__ == "__main__":
    do_predict(args)
