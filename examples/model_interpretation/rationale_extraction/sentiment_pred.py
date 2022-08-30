#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import re
import sys
import time
import logging
import json
import collections
from random import random
from tqdm import tqdm
from functools import reduce, partial
from pathlib import Path
import numpy as np
import logging
import argparse

import paddle
from paddlenlp.data import Stack, Tuple, Pad, Dict, Vocab
from paddlenlp.datasets import load_dataset, DatasetBuilder
from paddlenlp.transformers.roberta.tokenizer import RobertaTokenizer, RobertaBPETokenizer
from ernie.tokenizing_ernie import ErnieTokenizer

sys.path.append('../task/senti')
from LIME.lime_text import LimeTextExplainer
from rnn.model import LSTMModel, SelfInteractiveAttention, BiLSTMAttentionModel
from rnn.utils import CharTokenizer, convert_example
from saliency_map.utils import create_if_not_exists, get_warmup_and_linear_decay

sys.path.append('..')
from roberta.modeling import RobertaForSequenceClassification

sys.path.remove('..')
sys.path.remove('../task/senti')
sys.path.append('../..')
from model_interpretation.utils import convert_tokenizer_res_to_old_version

sys.path.remove('../..')


def get_args():
    parser = argparse.ArgumentParser('sentiment analysis prediction')

    parser.add_argument('--base_model',
                        required=True,
                        choices=['roberta_base', 'roberta_large', 'lstm'])
    parser.add_argument('--from_pretrained',
                        type=str,
                        required=True,
                        help='pretrained model directory or tag')
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=128,
                        help='max sentence length, should not greater than 512')
    parser.add_argument('--batch_size', type=int, default=1, help='batchsize')
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help='data directory includes train / develop data')
    parser.add_argument('--eval', action='store_true')

    parser.add_argument('--init_checkpoint',
                        type=str,
                        default=None,
                        help='checkpoint to warm start from')
    parser.add_argument('--wd',
                        type=float,
                        default=0.01,
                        help='weight decay, aka L2 regularizer')
    parser.add_argument(
        '--use_amp',
        action='store_true',
        help=
        'only activate AMP(auto mixed precision accelatoin) on TensorCore compatible devices'
    )
    parser.add_argument('--inter_mode',
                        type=str,
                        default="attention",
                        choices=[
                            "attention", "simple_gradient", "smooth_gradient",
                            "integrated_gradient", "lime"
                        ],
                        help='appoint the mode of interpretable.')
    parser.add_argument(
        '--n-samples',
        type=int,
        default=25,
        help='number of samples used for smooth gradient method')
    parser.add_argument('--output_dir',
                        type=Path,
                        required=True,
                        help='interpretable output directory')
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument("--vocab_path", type=str)
    parser.add_argument('--language',
                        type=str,
                        required=True,
                        help='Language that the model is built for')
    args = parser.parse_args()
    return args


class SentiData(DatasetBuilder):

    def _read(self, filename, language):
        with open(filename, "r", encoding="utf8") as f:
            for line in f.readlines():
                line_split = json.loads(line)
                yield {'id': line_split['id'], 'context': line_split['context']}


def create_dataloader(dataset,
                      trans_fn=None,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None):
    """
    Creats dataloader.

    Args:
        dataset(obj:`paddle.io.Dataset`): Dataset instance.
        trans_fn(obj:`callable`, optional, defaults to `None`): function to convert a data sample to input ids, etc.
        mode(obj:`str`, optional, defaults to obj:`train`): If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): The sample number of a mini-batch.
        batchify_fn(obj:`callable`, optional, defaults to `None`): function to generate mini-batch data by merging
            the sample list, None for only stack each fields of sample in axis
            0(same as :attr::`np.stack(..., axis=0)`).

    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == "train":
        sampler = paddle.io.DistributedBatchSampler(dataset=dataset,
                                                    batch_size=batch_size,
                                                    shuffle=shuffle)
    else:
        sampler = paddle.io.BatchSampler(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle)
    dataloader = paddle.io.DataLoader(dataset,
                                      batch_sampler=sampler,
                                      collate_fn=batchify_fn)
    return dataloader


def map_fn_senti(examples, tokenizer, language):
    print('load data %d' % len(examples))

    contexts = [example['context'] for example in examples]
    tokenized_examples = tokenizer(contexts, max_seq_len=args.max_seq_len)
    tokenized_examples = convert_tokenizer_res_to_old_version(
        tokenized_examples)

    return tokenized_examples


def truncate_offset(seg, start_offset, end_offset):
    seg_len = len(seg)
    for n in range(len(start_offset) - 1, -1, -1):
        if start_offset[n] < seg_len:
            end_offset[n] = seg_len
            break
        start_offset.pop(n)
        end_offset.pop(n)


def init_lstm_var(args):
    vocab = Vocab.load_vocabulary(args.vocab_path,
                                  unk_token='[UNK]',
                                  pad_token='[PAD]')
    tokenizer = CharTokenizer(vocab, args.language, '../punctuations')
    padding_idx = vocab.token_to_idx.get('[PAD]', 0)

    trans_fn = partial(convert_example,
                       tokenizer=tokenizer,
                       is_test=True,
                       language=args.language)

    #init attention layer
    lstm_hidden_size = 196
    attention = SelfInteractiveAttention(hidden_size=2 * lstm_hidden_size)
    model = BiLSTMAttentionModel(attention_layer=attention,
                                 vocab_size=len(tokenizer.vocab),
                                 lstm_hidden_size=lstm_hidden_size,
                                 num_classes=2,
                                 padding_idx=padding_idx)

    # Reads data and generates mini-batches.
    dev_ds = SentiData().read(os.path.join(args.data_dir, 'dev'), args.language)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=padding_idx),  # input_ids
        Stack(dtype="int64"),  # seq len
    ): [data for data in fn(samples)]

    dev_loader = create_dataloader(dev_ds,
                                   trans_fn=trans_fn,
                                   batch_size=args.batch_size,
                                   mode='validation',
                                   batchify_fn=batchify_fn)

    return model, tokenizer, dev_loader


def init_roberta_var(args):
    tokenizer = None
    if args.language == "ch":
        tokenizer = RobertaTokenizer.from_pretrained(args.from_pretrained)
    else:
        tokenizer = RobertaBPETokenizer.from_pretrained(args.from_pretrained)
    model = RobertaForSequenceClassification.from_pretrained(
        args.from_pretrained,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        dropout=0,
        num_labels=2,
        name='',
        return_inter_score=True)

    map_fn = partial(map_fn_senti, tokenizer=tokenizer, language=args.language)

    dev_ds = SentiData().read(os.path.join(args.data_dir, 'dev'), args.language)
    dev_ds.map(map_fn, batched=True)
    dev_batch_sampler = paddle.io.BatchSampler(dev_ds,
                                               batch_size=args.batch_size,
                                               shuffle=False)
    batchify_fn = lambda samples, fn=Dict(
        {
            "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
            "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id)
        }): fn(samples)

    dataloader = paddle.io.DataLoader(dataset=dev_ds,
                                      batch_sampler=dev_batch_sampler,
                                      collate_fn=batchify_fn,
                                      return_list=True)

    return model, tokenizer, dataloader


if __name__ == "__main__":
    args = get_args()
    if args.base_model.startswith('roberta'):
        model, tokenizer, dataloader = init_roberta_var(args)

    elif args.base_model == 'lstm':
        model, tokenizer, dataloader = init_lstm_var(args)
    else:
        raise ValueError('unsupported base model name.')

    with paddle.amp.auto_cast(enable=args.use_amp), \
        open(str(args.output_dir)+'/dev', 'w') as out_handle:
        # Load model
        sd = paddle.load(args.init_checkpoint)
        model.set_dict(sd)
        model.train()  # 为了取梯度，加载模型时dropout设为0
        print('load model from %s' % args.init_checkpoint)

        get_sub_word_ids = lambda word: map(
            str, tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)))

        for step, d in tqdm(enumerate(dataloader)):
            if step + 1 < args.start_id:
                continue

            result = {}
            if args.base_model.startswith('roberta'):
                input_ids, token_type_ids = d
                fwd_args = [input_ids, token_type_ids]
                fwd_kwargs = {}

                tokens = tokenizer.convert_ids_to_tokens(
                    input_ids[0, 1:-1].tolist())  # list

            elif args.base_model == 'lstm':
                input_ids, seq_lens = d
                fwd_args = [input_ids, seq_lens]
                fwd_kwargs = {}
                tokens = [
                    tokenizer.vocab.idx_to_token[input_id]
                    for input_id in input_ids.tolist()[0]
                ]

            result['id'] = dataloader.dataset.data[step]['id']

            probs, atts, embedded = model.forward_interpet(
                *fwd_args, **fwd_kwargs)
            pred_label = paddle.argmax(probs, axis=-1).tolist()[0]

            result['pred_label'] = pred_label
            result['probs'] = [
                float(format(prob, '.5f'))
                for prob in probs.numpy()[0].tolist()
            ]
            if args.language == 'en':
                result['context'] = tokenizer.convert_tokens_to_string(tokens)
            else:
                result['context'] = ''.join(tokens)
            out_handle.write(json.dumps(result, ensure_ascii=False) + '\n')
