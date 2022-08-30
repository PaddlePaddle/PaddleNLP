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
import re
import sys
import time
import json
import logging
import collections
import logging
import argparse

from random import random
from tqdm import tqdm
from functools import reduce, partial
from pathlib import Path
import numpy as np
import paddle
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from LIME.lime_text import LimeTextExplainer
from rnn.model import LSTMModel, SelfInteractiveAttention, BiLSTMAttentionModel
from rnn.utils import CharTokenizer, convert_example
from saliency_map.utils import create_if_not_exists, get_warmup_and_linear_decay
from paddlenlp.data import Dict, Pad, Stack, Vocab
from paddlenlp.datasets import DatasetBuilder
from paddlenlp.transformers.roberta.tokenizer import RobertaTokenizer, RobertaBPETokenizer
from ernie.tokenizing_ernie import ErnieTokenizer

from roberta.modeling import RobertaForSequenceClassification

sys.path.append('../../..')
from model_interpretation.utils import convert_tokenizer_res_to_old_version, match

sys.path.remove('../../..')

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser('interpret sentiment analysis task')
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
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--language',
                        type=str,
                        required=True,
                        help='language that the model is built for')
    args = parser.parse_args()
    return args


class Senti_data(DatasetBuilder):

    def _read(self, filename):
        with open(filename, "r", encoding="utf8") as f:
            for line in f.readlines():
                line_split = json.loads(line)
                yield {
                    'id': line_split['id'],
                    'context': line_split['context'],
                    'sent_token': line_split['sent_token']
                }


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


def map_fn_senti(examples, tokenizer, args):
    log.debug('load data %d' % len(examples))
    if args.language == 'en':
        contexts = [
            example['context'].encode('ascii', errors='replace').decode('UTF-8')
            for example in examples
        ]
    else:
        contexts = [example['context'] for example in examples]
    tokenized_examples = tokenizer(contexts, max_seq_len=args.max_seq_len)
    tokenized_examples = convert_tokenizer_res_to_old_version(
        tokenized_examples)
    for i in range(len(tokenized_examples)):
        tokenized_examples[i]['offset_mapping'] = [
            (0, 0)
        ] + tokenizer.get_offset_mapping(
            contexts[i])[:args.max_seq_len - 2] + [(0, 0)]
    return tokenized_examples


def init_lstm_var(args):
    vocab = Vocab.load_vocabulary(args.vocab_path,
                                  unk_token='[UNK]',
                                  pad_token='[PAD]')
    tokenizer = CharTokenizer(vocab, args.language, '../../punctuations')
    padding_idx = vocab.token_to_idx.get('[PAD]', 0)

    trans_fn = partial(convert_example,
                       tokenizer=tokenizer,
                       is_test=True,
                       language=args.language)

    # Init attention layer
    lstm_hidden_size = 196
    attention = SelfInteractiveAttention(hidden_size=2 * lstm_hidden_size)
    model = BiLSTMAttentionModel(attention_layer=attention,
                                 vocab_size=len(tokenizer.vocab),
                                 lstm_hidden_size=lstm_hidden_size,
                                 num_classes=2,
                                 padding_idx=padding_idx)

    # Reads data and generates mini-batches.
    dev_ds = Senti_data().read(args.data_dir)
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

    map_fn = partial(map_fn_senti, tokenizer=tokenizer, args=args)

    dev_ds = Senti_data().read(args.data_dir)
    dev_ds.map(map_fn, batched=True)
    dev_batch_sampler = paddle.io.BatchSampler(dev_ds,
                                               batch_size=args.batch_size,
                                               shuffle=False)
    batchify_fn = lambda samples, fn=Dict(
        {
            "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
            "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
            "offset_mapping": Pad(axis=0, pad_val=tokenizer.pad_token_id)
        }): fn(samples)

    dataloader = paddle.io.DataLoader(dataset=dev_ds,
                                      batch_sampler=dev_batch_sampler,
                                      collate_fn=batchify_fn,
                                      return_list=True)

    return model, tokenizer, dataloader


def extract_attention_scores(args, atts, input_ids, tokens, sub_word_id_dict,
                             result, offset, out_handle):
    if args.base_model.startswith('roberta'):
        inter_score = atts[-1][:, :, 0, :].mean(1)  # (bsz, seq)
        inter_score = inter_score[0][1:-1]  # remove CLS and SEP
        input_ids = input_ids[0][1:-1]

    elif args.base_model == 'lstm':
        inter_score = atts[0]
        input_ids = input_ids[0]

    length = (inter_score > 0).cast('int32').sum(-1).tolist()[0]
    assert len(tokens) == length, f"%s: {len(tokens)} != {length}" % (step + 1)

    char_attribution_dict = {}
    # Collect scores in different situation
    if args.base_model.startswith('roberta'):
        assert len(inter_score) == len(offset), str(
            len(inter_score)) + "not equal to" + str(len(offset))
        sorted_token = []
        for i in range(len(inter_score)):
            sorted_token.append([i, offset[i], inter_score[i]])

        char_attribution_dict = match(result['context'], result['sent_token'],
                                      sorted_token)

        result['char_attri'] = collections.OrderedDict()
        for token_info in sorted(char_attribution_dict,
                                 key=lambda x: x[2],
                                 reverse=True):
            result['char_attri'][str(
                token_info[0])] = [str(token_info[1]),
                                   float(token_info[2])]
        result.pop('sent_token')
    else:
        if args.language == 'ch':
            idx = 0
            for token, score in zip(tokens, inter_score.numpy().tolist()):
                char_attribution_dict[idx] = (token, score)
                idx += 1
        else:
            idx = 0
            for word, sub_word_score in zip(tokens, inter_score.tolist()):
                char_attribution_dict[idx] = (word, sub_word_score)
                idx += 1

        result['char_attri'] = collections.OrderedDict()
        for token_id, token_info in sorted(char_attribution_dict.items(),
                                           key=lambda x: x[1][1],
                                           reverse=True):
            result['char_attri'][token_id] = token_info

    out_handle.write(json.dumps(result, ensure_ascii=False) + '\n')


def extract_integrated_gradient_scores(args, atts, input_ids, tokens,
                                       sub_word_id_dict, fwd_args, fwd_kwargs,
                                       model, result, pred_label, err_total,
                                       offset, out_handle):
    embedded_grads_list = []
    for i in range(args.n_samples):
        probs, _, embedded = model.forward_interpet(*fwd_args,
                                                    **fwd_kwargs,
                                                    noise='integrated',
                                                    i=i,
                                                    n_samples=args.n_samples)
        predicted_class_prob = probs[0][pred_label]
        predicted_class_prob.backward(retain_graph=False)
        embedded_grad = embedded.grad
        model.clear_gradients()
        embedded_grads_list.append(embedded_grad)

        if i == 0:
            baseline_pred_confidence = probs.tolist()[0][pred_label]  # scalar
            baseline_embedded = embedded  # Tensor(1, seq_len, embed_size)
        elif i == args.n_samples - 1:
            pred_confidence = probs.tolist()[0][pred_label]  # scalar
            pred_embedded = embedded  # Tensor(1, seq_len, embed_size)

    embedded_grads_tensor = paddle.to_tensor(embedded_grads_list,
                                             dtype='float32',
                                             place=paddle.CUDAPlace(0),
                                             stop_gradient=True)

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
    log.debug(print_str % print_vals)

    inter_score.stop_gradient = True

    char_attribution_dict = {}
    if args.base_model.startswith('roberta'):
        inter_score = inter_score[0][1:-1]
        sorted_token = []
        for i in range(len(inter_score)):
            sorted_token.append([i, offset[i], inter_score[i]])
        char_attribution_dict = match(result['context'], result['sent_token'],
                                      sorted_token)

        result['char_attri'] = collections.OrderedDict()
        for token_info in sorted(char_attribution_dict,
                                 key=lambda x: x[2],
                                 reverse=True):
            result['char_attri'][str(
                token_info[0])] = [str(token_info[1]),
                                   float(token_info[2])]
        result.pop('sent_token')

    elif args.base_model == 'lstm':
        inter_score = inter_score[0]
        idx = 0
        for word, sub_word_score in zip(tokens, inter_score.tolist()):
            char_attribution_dict[idx] = (word, sub_word_score)
            idx += 1

        result['char_attri'] = collections.OrderedDict()
        for token_id, token_info in sorted(char_attribution_dict.items(),
                                           key=lambda x: x[1][1],
                                           reverse=True):
            result['char_attri'][token_id] = token_info

    out_handle.write(json.dumps(result, ensure_ascii=False) + '\n')
    return err_total


def extract_LIME_scores(args, tokenizer, tokens, pred_label, model, probs,
                        result, lime_err_total, lime_score_total,
                        lime_relative_err_total, out_handle):
    explainer = LimeTextExplainer(class_names=['neg', 'pos'],
                                  verbose=False,
                                  language=args.language)

    if_lstm = (args.base_model == 'lstm')
    explain_res = None

    text_instance = result['context']

    explain_res = explainer.explain_instance(
        text_instance=text_instance,
        tokenizer=tokenizer,
        pred_label=pred_label,
        classifier_fn=model.forward_interpet,
        num_samples=5000,
        if_lstm=if_lstm)

    exp, indexed_string, relative_err, err = explain_res

    score = exp.score[pred_label]
    local_exps = exp.local_exp
    ridge_pred = exp.local_pred[pred_label]
    model_pred = probs.numpy().tolist()[0][pred_label]

    lime_score_total.append(score)
    lime_relative_err_total.append(relative_err)
    lime_err_total.append(err)
    log.debug('score: %.2f' % score)
    log.debug('relative_err: %.2f' % relative_err)
    log.debug('err: %.2f' % err)
    log.debug('ridge_pred: %.2f\tpred: %.2f\tdelta: %.2f' %
              (ridge_pred, model_pred, ridge_pred - model_pred))

    for kind, local_exp in local_exps.items():  #only have one iteration here
        char_attribution_dict = []

        for idx in range(len(result['sent_token'])):
            t = result['sent_token'][idx]  #.replace('Ġ', '')
            got_score = False
            for word_id, attribution in local_exp:
                if indexed_string.inverse_vocab[word_id] == t:
                    char_attribution_dict.append((idx, t, attribution))
                    got_score = True
                    break
            if not got_score:
                char_attribution_dict.append((idx, t, 0))
        char_attribution_dict = sorted(char_attribution_dict,
                                       key=lambda x: x[2],
                                       reverse=True)

        result['char_attri'] = collections.OrderedDict()
        for s in char_attribution_dict:
            result['char_attri'][s[0]] = (s[1], s[2])

    out_handle.write(json.dumps(result, ensure_ascii=False) + '\n')
    return lime_err_total, lime_score_total, lime_relative_err_total


if __name__ == "__main__":
    args = get_args()
    if args.base_model.startswith('roberta'):
        model, tokenizer, dataloader = init_roberta_var(args)
    elif args.base_model == 'lstm':
        model, tokenizer, dataloader = init_lstm_var(args)
    else:
        raise ValueError('unsupported base model name.')

    assert args.eval, 'INTERPRETER must be run in eval mode'
    with paddle.amp.auto_cast(enable=args.use_amp), \
        open(os.path.join(args.output_dir, 'interpret' + f'.{args.inter_mode}'), 'w') as out_handle:

        # Load model
        sd = paddle.load(args.init_checkpoint)
        model.set_dict(sd)
        model.train()  # set dropout to 0 in order to get the gradient
        log.debug('load model from %s' % args.init_checkpoint)

        get_sub_word_ids = lambda word: map(
            str, tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)))
        for step, d in tqdm(enumerate(dataloader)):
            if step + 1 < args.start_id:  #start from the step's instance
                continue
            # Initialize input_ids, fwd_args, tokens
            result = {}
            offset = None
            if args.base_model.startswith('roberta'):
                input_ids, token_type_ids, offset_map = d
                fwd_args = [input_ids, token_type_ids]
                fwd_kwargs = {}
                tokens = tokenizer.convert_ids_to_tokens(
                    input_ids[0, 1:-1].tolist())  # list
                offset = offset_map[0, 1:-1]

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
            sub_word_id_dict = []
            err_total = []
            lime_err_total, lime_score_total, lime_relative_err_total = [], [], []

            result['context'] = dataloader.dataset.data[step]['context']
            result['sent_token'] = dataloader.dataset.data[step]['sent_token']

            # Attention
            if args.inter_mode == "attention":
                #extract attention scores and write resutls to file
                extract_attention_scores(args, atts, input_ids, tokens,
                                         sub_word_id_dict, result, offset,
                                         out_handle)

            # Integrated_gradient
            elif args.inter_mode == 'integrated_gradient':
                err_total = extract_integrated_gradient_scores(
                    args, atts, input_ids, tokens, sub_word_id_dict, fwd_args,
                    fwd_kwargs, model, result, pred_label, err_total, offset,
                    out_handle)

            # LIME
            elif args.inter_mode == 'lime':
                lime_err_total, lime_score_total, lime_relative_err_total = extract_LIME_scores(
                    args, tokenizer, tokens, pred_label, model, probs, result,
                    lime_err_total, lime_score_total, lime_relative_err_total,
                    out_handle)

            else:
                raise KeyError(f"Unkonwn interpretable mode: {args.inter_mode}")

        if args.inter_mode == 'lime':
            log.debug(np.average(np.array(lime_relative_err_total)))
