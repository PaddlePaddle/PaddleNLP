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

sys.path.append('../task/similarity')
from LIME.lime_text import LimeTextExplainer

sys.path.append('..')
from roberta.modeling import RobertaForSequenceClassification

sys.path.remove('..')
from simnet.utils import CharTokenizer, preprocess_data
from simnet.model import SimNet

sys.path.remove('../task/similarity')
sys.path.append('../..')
from model_interpretation.utils import convert_tokenizer_res_to_old_version

sys.path.remove('../..')


def get_args():
    parser = argparse.ArgumentParser('textual similarity prediction')

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
    parser.add_argument('--output_dir',
                        type=Path,
                        required=True,
                        help='interpretable output directory')
    parser.add_argument('--language', type=str, required=True)
    args = parser.parse_args()
    return args


class SimilarityData(DatasetBuilder):

    def _read(self, filename):
        with open(filename, "r", encoding="utf8") as f:
            for line in f.readlines():
                line_split = json.loads(line)
                if args.language == 'ch':
                    yield {
                        'id': line_split['id'],
                        'query': line_split['context'][0],
                        'title': line_split['context'][1]
                    }
                else:
                    yield {
                        'id': line_split['id'],
                        'sentence1': line_split['context'][0],
                        'sentence2': line_split['context'][1]
                    }


def map_fn_senti(examples, tokenizer):
    print('load data %d' % len(examples))
    if args.language == 'ch':
        query = 'query'
        title = 'title'
    else:
        query = 'sentence1'
        title = 'sentence2'
    queries = [example[query] for example in examples]
    titles = [example[title] for example in examples]
    tokenized_examples = tokenizer(queries,
                                   titles,
                                   max_seq_len=args.max_seq_len)
    tokenized_examples = convert_tokenizer_res_to_old_version(
        tokenized_examples)

    return tokenized_examples


def init_roberta_var(args):
    if args.language == 'ch':
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

    map_fn = partial(map_fn_senti, tokenizer=tokenizer)

    dev_ds = SimilarityData().read(os.path.join(args.data_dir, 'dev'))
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

    return model, tokenizer, dataloader, dev_ds


def init_lstm_var(args):
    if args.language == 'ch':
        vocab = Vocab.load_vocabulary("../task/similarity/simnet/vocab.char",
                                      unk_token='[UNK]',
                                      pad_token='[PAD]')
    else:
        vocab = Vocab.load_vocabulary("../task/similarity/simnet/vocab_QQP",
                                      unk_token='[UNK]',
                                      pad_token='[PAD]')

    tokenizer = CharTokenizer(vocab, args.language, '../punctuations')
    model = SimNet(network='lstm', vocab_size=len(vocab), num_classes=2)

    dev_ds = SimilarityData().read(os.path.join(args.data_dir, 'dev'))
    dev_examples = preprocess_data(dev_ds.data,
                                   tokenizer,
                                   language=args.language)
    batches = [
        dev_examples[idx:idx + args.batch_size]
        for idx in range(0, len(dev_examples), args.batch_size)
    ]

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=vocab.token_to_idx.get('[PAD]', 0)),  # query_ids
        Pad(axis=0, pad_val=vocab.token_to_idx.get('[PAD]', 0)),  # title_ids
        Stack(dtype="int64"),  # query_seq_lens
        Stack(dtype="int64"),  # title_seq_lens
    ): [data for data in fn(samples)]

    return model, tokenizer, batches, batchify_fn, vocab, dev_ds


if __name__ == "__main__":
    args = get_args()
    if args.base_model.startswith('roberta'):
        model, tokenizer, dataloader, dev_ds = init_roberta_var(args)

    elif args.base_model == 'lstm':
        model, tokenizer, dataloader, batchify_fn, vocab, dev_ds = init_lstm_var(
            args)
    else:
        raise ValueError('unsupported base model name.')

    with paddle.amp.auto_cast(enable=args.use_amp), \
        open(str(args.output_dir)+'/dev', 'w') as out_handle:
        # Load model
        sd = paddle.load(args.init_checkpoint)
        model.set_dict(sd)
        model.train()  # 为了取梯度，加载模型时dropout设为0
        print('load model from %s' % args.init_checkpoint)

        for step, d in tqdm(enumerate(dataloader)):

            result = {}
            if args.base_model.startswith('roberta'):
                input_ids, token_type_ids = d
                fwd_args = [input_ids, token_type_ids]
                fwd_kwargs = {}

                SEP_idx = input_ids.tolist()[0].index(tokenizer.sep_token_id)
                q_tokens = tokenizer.convert_ids_to_tokens(
                    input_ids[0, 1:SEP_idx].tolist())  # list
                if args.language == 'ch':
                    t_tokens = tokenizer.convert_ids_to_tokens(
                        input_ids[0, SEP_idx + 1:-1].tolist())  # list
                else:
                    t_tokens = tokenizer.convert_ids_to_tokens(
                        input_ids[0, SEP_idx + 2:-1].tolist())  # list

            elif args.base_model == 'lstm':
                query_ids, title_ids, query_seq_lens, title_seq_lens = batchify_fn(
                    d)
                query_ids = paddle.to_tensor(query_ids)
                title_ids = paddle.to_tensor(title_ids)
                query_seq_lens = paddle.to_tensor(query_seq_lens)
                title_seq_lens = paddle.to_tensor(title_seq_lens)

                fwd_args = [
                    query_ids, title_ids, query_seq_lens, title_seq_lens
                ]
                fwd_kwargs = {}
                q_tokens = [
                    vocab._idx_to_token[idx] for idx in query_ids.tolist()[0]
                ]
                t_tokens = [
                    vocab._idx_to_token[idx] for idx in title_ids.tolist()[0]
                ]

            result['id'] = dev_ds.data[step]['id']

            probs, atts, embedded = model.forward_interpret(
                *fwd_args, **fwd_kwargs)
            pred_label = paddle.argmax(probs, axis=-1).tolist()[0]

            result['pred_label'] = pred_label
            result['probs'] = [
                float(format(prob, '.5f'))
                for prob in probs.numpy()[0].tolist()
            ]
            if args.language == 'ch':
                result['query'] = ''.join(q_tokens)
                result['title'] = ''.join(t_tokens)
            else:
                result['query'] = tokenizer.convert_tokens_to_string(q_tokens)
                result['title'] = tokenizer.convert_tokens_to_string(t_tokens)

            out_handle.write(json.dumps(result, ensure_ascii=False) + '\n')
