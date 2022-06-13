# !/usr/bin/env python3
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
import sys
import re
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
from paddlenlp.data import Stack, Tuple, Pad, Vocab
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Dict, Pad, Stack
from paddlenlp.datasets import DatasetBuilder
from paddlenlp.transformers.roberta.tokenizer import RobertaTokenizer, RobertaBPETokenizer

from roberta.modeling import RobertaForSequenceClassification
from simnet.utils import CharTokenizer, preprocess_data
from simnet.model import SimNet
from LIME.lime_text import LimeTextExplainer

sys.path.append('../../..')
from model_interpretation.utils import convert_tokenizer_res_to_old_version, match

sys.path.remove('../../..')

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser('interpret textual similarity task')
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
    parser.add_argument('--language',
                        type=str,
                        required=True,
                        help='Language that the model is based on')
    args = parser.parse_args()
    return args


class Similarity_data(DatasetBuilder):

    def _read(self, filename):
        with open(filename, "r", encoding="utf8") as f:
            for line in f.readlines():
                line_split = json.loads(line)
                if args.language == 'ch':
                    yield {
                        'id': line_split['id'],
                        'query': line_split['query'],
                        'title': line_split['title'],
                        'text_q_seg': line_split['text_q_seg'],
                        'text_t_seg': line_split['text_t_seg']
                    }
                else:
                    yield {
                        'id': line_split['id'],
                        'sentence1': line_split['sentence1'],
                        'sentence2': line_split['sentence2'],
                        'text_q_seg': line_split['text_q_seg'],
                        'text_t_seg': line_split['text_t_seg']
                    }


def map_fn_senti(examples, tokenizer, language):
    print('load data %d' % len(examples))
    if language == 'ch':
        q_name = "query"
        t_name = "title"
        queries = [example[q_name] for example in examples]
        titles = [example[t_name] for example in examples]
    else:
        q_name = "sentence1"
        t_name = "sentence2"
        queries = [
            example[q_name].encode('ascii', errors='replace').decode('UTF-8')
            for example in examples
        ]
        titles = [
            example[t_name].encode('ascii', errors='replace').decode('UTF-8')
            for example in examples
        ]
    tokenized_examples = tokenizer(queries,
                                   titles,
                                   max_seq_len=args.max_seq_len)

    tokenized_examples = convert_tokenizer_res_to_old_version(
        tokenized_examples)

    for i in range(len(tokenized_examples)):
        tokenized_examples[i]['query_offset_mapping'] = [
            (0, 0)
        ] + tokenizer.get_offset_mapping(
            queries[i])[:args.max_seq_len - 2] + [(0, 0)]
        tokenized_examples[i]['title_offset_mapping'] = [
            (0, 0)
        ] + tokenizer.get_offset_mapping(
            titles[i])[:args.max_seq_len - 2] + [(0, 0)]

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

    map_fn = partial(map_fn_senti, tokenizer=tokenizer, language=args.language)

    dev_ds = Similarity_data().read(args.data_dir)
    dev_ds.map(map_fn, batched=True)
    dev_batch_sampler = paddle.io.BatchSampler(dev_ds,
                                               batch_size=args.batch_size,
                                               shuffle=False)
    batchify_fn = lambda samples, fn=Dict(
        {
            "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
            "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
            "query_offset_mapping": Pad(axis=0, pad_val=tokenizer.pad_token_id),
            "title_offset_mapping": Pad(axis=0, pad_val=tokenizer.pad_token_id)
        }): fn(samples)

    dataloader = paddle.io.DataLoader(dataset=dev_ds,
                                      batch_sampler=dev_batch_sampler,
                                      collate_fn=batchify_fn,
                                      return_list=True)

    return model, tokenizer, dataloader, dev_ds


def init_lstm_var(args):
    if args.language == 'ch':
        vocab = Vocab.load_vocabulary("simnet/vocab.char",
                                      unk_token='[UNK]',
                                      pad_token='[PAD]')
    else:
        vocab = Vocab.load_vocabulary("simnet/vocab_QQP",
                                      unk_token='[UNK]',
                                      pad_token='[PAD]')

    tokenizer = CharTokenizer(vocab, args.language, '../../punctuations')
    model = SimNet(network='lstm', vocab_size=len(vocab), num_classes=2)

    dev_ds = Similarity_data().read(args.data_dir)
    dev_examples = preprocess_data(dev_ds.data, tokenizer, args.language)
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


def get_seq_token_num(language):
    if language == 'ch':
        add_idx = 1
    else:
        add_idx = 2
    return add_idx


def get_qt_tokens(base_model,
                  d,
                  add_idx=None,
                  tokenizer=None,
                  batchify_fn=None,
                  vocab=None):
    SEP_idx = 0
    if base_model == 'roberta':
        input_ids, token_type_ids, query_offset_map, title_offset_map = d
        fwd_args = [input_ids, token_type_ids]
        fwd_kwargs = {}

        SEP_idx = input_ids.tolist()[0].index(tokenizer.sep_token_id)
        q_tokens = tokenizer.convert_ids_to_tokens(
            input_ids[0, 1:SEP_idx].tolist())  # list
        t_tokens = tokenizer.convert_ids_to_tokens(
            input_ids[0, SEP_idx + add_idx:-1].tolist())  # list
        q_offset = query_offset_map[0, 1:-1].tolist()
        t_offset = title_offset_map[0, 1:-1].tolist()
        return q_tokens, t_tokens, SEP_idx, fwd_args, fwd_kwargs, q_offset, t_offset

    if base_model == 'lstm':
        query_ids, title_ids, query_seq_lens, title_seq_lens = batchify_fn(d)
        query_ids = paddle.to_tensor(query_ids)
        title_ids = paddle.to_tensor(title_ids)
        query_seq_lens = paddle.to_tensor(query_seq_lens)
        title_seq_lens = paddle.to_tensor(title_seq_lens)

        fwd_args = [query_ids, title_ids, query_seq_lens, title_seq_lens]
        fwd_kwargs = {}
        q_tokens = [vocab._idx_to_token[idx] for idx in query_ids.tolist()[0]]
        t_tokens = [vocab._idx_to_token[idx] for idx in title_ids.tolist()[0]]
        return q_tokens, t_tokens, SEP_idx, fwd_args, fwd_kwargs


def extract_attention_scores(args, result, atts, q_tokens, t_tokens, out_handle,
                             SEP_idx, q_offset, t_offset, add_idx):
    if args.base_model.startswith('roberta'):
        inter_score = atts[-1][:, :, 0, :].mean(1)  # (bsz, seq)
        q_inter_score = inter_score[0][1:SEP_idx]  # remove CLS and SEP
        t_inter_score = inter_score[0][SEP_idx +
                                       add_idx:-1]  # remove CLS and SEP
    elif args.base_model == 'lstm':
        q_inter_score = atts[0][0]
        t_inter_score = atts[1][0]

    q_length = (q_inter_score > 0).cast('int32').sum(-1)[0]
    t_length = (t_inter_score > 0).cast('int32').sum(-1)[0]
    assert len(q_tokens) == q_length, f"{len(q_tokens)} != {q_length}"
    assert len(t_tokens) == t_length, f"{len(t_tokens)} != {t_length}"

    q_char_attribution_dict, t_char_attribution_dict = {}, {}
    if args.base_model.startswith('roberta'):
        # Query
        sorted_token = []
        for i in range(len(q_inter_score)):
            sorted_token.append([i, q_offset[i], q_inter_score[i]])
        q_char_attribution_dict = match(result['query'], result['text_q_seg'],
                                        sorted_token)
        result['query_char_attri'] = collections.OrderedDict()
        for token_info in sorted(q_char_attribution_dict,
                                 key=lambda x: x[2],
                                 reverse=True):
            result['query_char_attri'][str(
                token_info[0])] = [str(token_info[1]),
                                   float(token_info[2])]
        result.pop('text_q_seg')

        #Title
        sorted_token = []
        for i in range(len(t_inter_score)):
            sorted_token.append([i, t_offset[i], t_inter_score[i]])
        t_char_attribution_dict = match(result['title'], result['text_t_seg'],
                                        sorted_token)
        result['title_char_attri'] = collections.OrderedDict()
        for token_info in sorted(t_char_attribution_dict,
                                 key=lambda x: x[2],
                                 reverse=True):
            result['title_char_attri'][str(
                token_info[0])] = [str(token_info[1]),
                                   float(token_info[2])]
        result.pop('text_t_seg')

    else:
        idx = 0
        for token, score in zip(q_tokens, q_inter_score.tolist()):
            q_char_attribution_dict[idx] = (token, score)
            idx += 1
        for token, score in zip(t_tokens, t_inter_score.tolist()):
            t_char_attribution_dict[idx] = (token, score)
            idx += 1

        result['query_char_attri'], result[
            'title_char_attri'] = collections.OrderedDict(
            ), collections.OrderedDict()
        for token, attri in sorted(q_char_attribution_dict.items(),
                                   key=lambda x: x[1][1],
                                   reverse=True):
            result['query_char_attri'][token] = attri
        for token, attri in sorted(t_char_attribution_dict.items(),
                                   key=lambda x: x[1][1],
                                   reverse=True):
            result['title_char_attri'][token] = attri

    out_handle.write(json.dumps(result, ensure_ascii=False) + '\n')


def IG_roberta_inter_score(args, embedded_grads_list, pred_embedded,
                           baseline_embedded, pred_confidence,
                           baseline_pred_confidence, SEP_idx, add_idx,
                           err_total):
    embedded_grads_tensor = paddle.to_tensor(embedded_grads_list,
                                             dtype='float32',
                                             place=paddle.CUDAPlace(0),
                                             stop_gradient=True)

    # Tensor(n_samples-1, 1, seq_len, embed_size)
    trapezoidal_grads = (embedded_grads_tensor[1:] +
                         embedded_grads_tensor[:-1]) / 2
    integral_grads = trapezoidal_grads.sum(0) / trapezoidal_grads.shape[
        0]  # Tensor(1, seq_len, embed_size)
    inter_score = (pred_embedded - baseline_embedded
                   ) * integral_grads  # Tensor(1, seq_len, embed_size)
    inter_score = inter_score.sum(-1)  # Tensor(1, seq_len)

    # eval err
    delta_pred_confidence = pred_confidence - baseline_pred_confidence
    sum_gradient = inter_score.sum().tolist()[0]
    err = (delta_pred_confidence - sum_gradient +
           1e-12) / (delta_pred_confidence + 1e-12)
    err_total.append(np.abs(err))

    print_str = '%s\t%d\t%.3f\t%.3f\t%.3f\t%.3f'
    print_vals = (result['id'], args.n_samples, delta_pred_confidence,
                  sum_gradient, err, np.average(err_total))
    print(print_str % print_vals)

    inter_score.stop_gradient = True
    q_inter_score = inter_score[0][1:SEP_idx]  # remove CLS and SEP
    t_inter_score = inter_score[0][SEP_idx + add_idx:-1]  # remove CLS and SEP

    return q_inter_score, t_inter_score


def IG_lstm_inter_score(q_embedded_grads_list, pred_embedded, baseline_embedded,
                        idx):
    # query
    q_embedded_grads_tensor = paddle.to_tensor(q_embedded_grads_list,
                                               dtype='float32',
                                               place=paddle.CUDAPlace(0),
                                               stop_gradient=True)
    q_trapezoidal_grads = (q_embedded_grads_tensor[1:] +
                           q_embedded_grads_tensor[:-1]
                           ) / 2  # Tensor(n_samples-1, 1, seq_len, embed_size)
    q_integral_grads = q_trapezoidal_grads.sum(0) / q_trapezoidal_grads.shape[
        0]  # Tensor(1, seq_len, embed_size)
    q_inter_score = (pred_embedded[idx] - baseline_embedded[idx]
                     ) * q_integral_grads  # Tensor(1, seq_len, embed_size)
    q_inter_score = q_inter_score.sum(-1)  # Tensor(1, seq_len)
    q_inter_score.stop_gradient = True
    q_inter_score = q_inter_score[0]

    return q_inter_score


def extract_integrated_gradient_scores(args, result, fwd_args, fwd_kwargs,
                                       model, q_tokens, t_tokens, out_handle,
                                       SEP_idx, add_idx, q_offset, t_offset,
                                       err_total):
    embedded_grads_list = []
    q_embedded_grads_list, t_embedded_grads_list = [], []
    for i in range(args.n_samples):
        probs, _, embedded = model.forward_interpret(*fwd_args,
                                                     **fwd_kwargs,
                                                     noise='integrated',
                                                     i=i,
                                                     n_samples=args.n_samples)
        predicted_class_prob = probs[0][pred_label]
        predicted_class_prob.backward(retain_graph=False)

        if args.base_model.startswith('roberta'):
            embedded_grad = embedded.grad
            embedded_grads_list.append(embedded_grad)
        elif args.base_model == 'lstm':
            q_embedded, t_embedded = embedded
            q_embedded_grad = q_embedded.grad
            t_embedded_grad = t_embedded.grad
            q_embedded_grads_list.append(q_embedded_grad)
            t_embedded_grads_list.append(t_embedded_grad)
        model.clear_gradients()
        if i == 0:
            baseline_pred_confidence = probs.tolist()[0][pred_label]  # scalar
            baseline_embedded = embedded  # Tensor(1, seq_len, embed_size)
        elif i == args.n_samples - 1:
            pred_confidence = probs.tolist()[0][pred_label]  # scalar
            pred_embedded = embedded  # Tensor(1, seq_len, embed_size)

    if args.base_model.startswith('roberta'):
        q_inter_score, t_inter_score = IG_roberta_inter_score(
            args, embedded_grads_list, pred_embedded, baseline_embedded,
            pred_confidence, baseline_pred_confidence, SEP_idx, add_idx,
            err_total)
    elif args.base_model == 'lstm':
        q_inter_score = IG_lstm_inter_score(q_embedded_grads_list,
                                            pred_embedded, baseline_embedded, 0)
        t_inter_score = IG_lstm_inter_score(t_embedded_grads_list,
                                            pred_embedded, baseline_embedded, 1)

    q_char_attribution_dict, t_char_attribution_dict = {}, {}
    if args.base_model.startswith('roberta'):
        # Query
        sorted_token = []
        for i in range(len(q_inter_score)):
            sorted_token.append([i, q_offset[i], q_inter_score[i]])
        q_char_attribution_dict = match(result['query'], result['text_q_seg'],
                                        sorted_token)
        result['query_char_attri'] = collections.OrderedDict()
        for token_info in sorted(q_char_attribution_dict,
                                 key=lambda x: x[2],
                                 reverse=True):
            result['query_char_attri'][str(
                token_info[0])] = [str(token_info[1]),
                                   float(token_info[2])]
        result.pop('text_q_seg')

        #Title
        sorted_token = []
        for i in range(len(t_inter_score)):
            sorted_token.append([i, t_offset[i], t_inter_score[i]])
        t_char_attribution_dict = match(result['title'], result['text_t_seg'],
                                        sorted_token)
        result['title_char_attri'] = collections.OrderedDict()
        for token_info in sorted(t_char_attribution_dict,
                                 key=lambda x: x[2],
                                 reverse=True):
            result['title_char_attri'][str(
                token_info[0])] = [str(token_info[1]),
                                   float(token_info[2])]
        result.pop('text_t_seg')
    else:
        idx = 0
        for token, score in zip(q_tokens, q_inter_score.tolist()):
            q_char_attribution_dict[idx] = (token, score)
            idx += 1
        for token, score in zip(t_tokens, t_inter_score.tolist()):
            t_char_attribution_dict[idx] = (token, score)
            idx += 1

        result['query_char_attri'], result[
            'title_char_attri'] = collections.OrderedDict(
            ), collections.OrderedDict()
        for token, attri in sorted(q_char_attribution_dict.items(),
                                   key=lambda x: x[1][1],
                                   reverse=True):
            result['query_char_attri'][token] = attri
        for token, attri in sorted(t_char_attribution_dict.items(),
                                   key=lambda x: x[1][1],
                                   reverse=True):
            result['title_char_attri'][token] = attri

    out_handle.write(json.dumps(result, ensure_ascii=False) + '\n')


def extract_LIME_scores(args, q_tokens, t_tokens, result, tokenizer, pred_label,
                        fwd_args, fwd_kwargs, model, probs, out_handle):
    explainer = LimeTextExplainer(class_names=['neg', 'pos'],
                                  verbose=False,
                                  language=args.language)
    if_lstm = (args.base_model == 'lstm')

    explain_res_q = explainer.explain_instance(
        text_instance_q=result['query'],
        text_instance_t=result['title'],
        analysis_query=True,
        tokenizer=tokenizer,
        pred_label=pred_label,
        classifier_fn=model.forward_interpret,
        num_samples=5000,
        if_lstm=if_lstm)
    exp_q, indexed_string_q, relative_err, err = explain_res_q
    local_exps_q = exp_q.local_exp

    explain_res_t = explainer.explain_instance(
        text_instance_q=result['query'],
        text_instance_t=result['title'],
        analysis_query=False,
        tokenizer=tokenizer,
        pred_label=pred_label,
        classifier_fn=model.forward_interpret,
        num_samples=5000,
        if_lstm=if_lstm)
    exp_t, indexed_string_t, _, _ = explain_res_t
    local_exps_t = exp_t.local_exp

    # query
    char_attribution_dict = []
    for kind, local_exp in local_exps_q.items():
        for idx in range(len(result['text_q_seg'])):
            t = result['text_q_seg'][idx]  #.replace('Ġ', '')
            got_score = False
            for word_id, attribution in local_exp:
                if indexed_string_q.inverse_vocab[word_id] == t:
                    char_attribution_dict.append((idx, t, attribution))
                    got_score = True
                    break
                if not got_score:
                    char_attribution_dict.append((idx, t, 0))
    char_attribution_dict = sorted(char_attribution_dict,
                                   key=lambda x: x[2],
                                   reverse=True)
    result['query_char_attri'] = collections.OrderedDict()
    for s in char_attribution_dict:
        result['query_char_attri'][s[0]] = (s[1], s[2])

    # title
    char_attribution_dict = []
    for kind, local_exp in local_exps_t.items():
        for idx in range(len(result['text_t_seg'])):
            t = result['text_t_seg'][idx]  #.replace('Ġ', '')
            got_score = False
            for word_id, attribution in local_exp:
                if indexed_string_t.inverse_vocab[word_id] == t:
                    char_attribution_dict.append((idx, t, attribution))
                    got_score = True
                    break
                if not got_score:
                    char_attribution_dict.append((idx, t, 0))
    char_attribution_dict = sorted(char_attribution_dict,
                                   key=lambda x: x[2],
                                   reverse=True)
    result['title_char_attri'] = collections.OrderedDict()
    for s in char_attribution_dict:
        result['title_char_attri'][s[0]] = (s[1], s[2])

    out_handle.write(json.dumps(result, ensure_ascii=False) + '\n')
    return exp_q, exp_t, relative_err, err


def LIME_error_evaluation(exp_q, pred_label, probs, lime_score_total,
                          lime_relative_err_total, lime_err_total, relative_err,
                          err):
    # err evaluation
    score = exp_q.score[pred_label]
    ridge_pred = exp_q.local_pred[pred_label]
    model_pred = probs.numpy().tolist()[0][pred_label]

    lime_score_total.append(score)
    lime_relative_err_total.append(relative_err)
    lime_err_total.append(err)
    print('score: %.2f' % score)
    print('relative_err: %.2f' % relative_err)
    print('err: %.2f' % err)
    print('ridge_pred: %.2f\tpred: %.2f\tdelta: %.2f' %
          (ridge_pred, model_pred, ridge_pred - model_pred))
    return lime_score_total, lime_relative_err_total, lime_err_total


g_splitter = re.compile(r'([\u4e00-\u9fa5])')

if __name__ == "__main__":
    args = get_args()
    if args.base_model.startswith('roberta'):
        model, tokenizer, dataloader, dev_ds = init_roberta_var(args)
    elif args.base_model == 'lstm':
        model, tokenizer, dataloader, batchify_fn, vocab, dev_ds = init_lstm_var(
            args)
    else:
        raise ValueError('unsupported base model name.')

    assert args.eval, 'INTERPRETER must be run in eval mode'
    with paddle.amp.auto_cast(enable=args.use_amp), \
        open(os.path.join(args.output_dir, 'interpret' + f'.{args.inter_mode}'), 'w') as out_handle:
        # Load model
        sd = paddle.load(args.init_checkpoint)
        model.set_dict(sd)
        model.train(
        )  # Set dropout to 0 when init the model to collect the gradient
        print('load model from %s' % args.init_checkpoint)

        # For IG
        err_total = []
        # For LIME
        lime_score_total = []
        lime_relative_err_total = []
        lime_err_total = []
        # For Roberta
        sub_word_id_dict_query = []
        sub_word_id_dict_title = []
        # For LSTM
        q_offset, t_offset = None, None

        get_sub_word_ids = lambda word: map(
            str, tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)))
        for step, d in tqdm(enumerate(dataloader)):
            if step + 1 < args.start_id:
                continue

            result = {}
            # English and Chinese models have different numbers of [SEQ] tokens between query and title
            add_idx = get_seq_token_num(args.language)

            if args.base_model.startswith('roberta'):
                q_tokens, t_tokens, SEP_idx, fwd_args, fwd_kwargs, q_offset, t_offset = get_qt_tokens(
                    base_model='roberta',
                    d=d,
                    add_idx=add_idx,
                    tokenizer=tokenizer)
            elif args.base_model == 'lstm':
                q_tokens, t_tokens, SEP_idx, fwd_args, fwd_kwargs = get_qt_tokens(
                    base_model='lstm',
                    d=d,
                    batchify_fn=batchify_fn,
                    vocab=vocab)

            result['id'] = dev_ds.data[step]['id']
            result['text_q_seg'] = dev_ds.data[step]['text_q_seg']
            result['text_t_seg'] = dev_ds.data[step]['text_t_seg']

            probs, atts, embedded = model.forward_interpret(
                *fwd_args, **fwd_kwargs)
            pred_label = paddle.argmax(probs, axis=-1).tolist()[0]

            result['pred_label'] = pred_label
            result['probs'] = [
                float(format(prob, '.5f'))
                for prob in probs.numpy()[0].tolist()
            ]

            if args.language == 'ch':
                result['query'] = dev_ds.data[step]['query']
                result['title'] = dev_ds.data[step]['title']
            else:
                result['query'] = dev_ds.data[step]['sentence1']
                result['title'] = dev_ds.data[step]['sentence2']

            # Attention
            if args.inter_mode == "attention":
                extract_attention_scores(args, result, atts, q_tokens, t_tokens,
                                         out_handle, SEP_idx, q_offset,
                                         t_offset, add_idx)

            elif args.inter_mode == 'integrated_gradient':
                extract_integrated_gradient_scores(args, result, fwd_args,
                                                   fwd_kwargs, model, q_tokens,
                                                   t_tokens, out_handle,
                                                   SEP_idx, add_idx, q_offset,
                                                   t_offset, err_total)

            elif args.inter_mode == 'lime':
                exp_q, exp_t, relative_err, err = extract_LIME_scores(
                    args, q_tokens, t_tokens, result, tokenizer, pred_label,
                    fwd_args, fwd_kwargs, model, probs, out_handle)
                lime_score_total, lime_relative_err_total, lime_err_total = LIME_error_evaluation(
                    exp_q, pred_label, probs, lime_score_total,
                    lime_relative_err_total, lime_err_total, relative_err, err)

            else:
                raise KeyError(f"Unkonwn interpretable mode: {args.inter_mode}")

        if args.inter_mode == 'lime':
            print(np.average(np.array(lime_relative_err_total)))
