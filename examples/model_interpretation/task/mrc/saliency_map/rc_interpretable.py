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
import json
import collections
from tqdm import tqdm
from functools import partial
from pathlib import Path
import logging
import argparse

import paddle
from paddle.io import DataLoader
from paddlenlp.data import Stack, Pad, Dict
from paddlenlp.transformers import ErnieTokenizer
from squad import RCInterpret
from paddlenlp.transformers.roberta.tokenizer import RobertaTokenizer, RobertaBPETokenizer

from roberta.modeling import RobertaForQuestionAnswering

sys.path.append('../../..')
from model_interpretation.utils import convert_tokenizer_res_to_old_version, match

sys.path.remove('../../..')

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser('mrc task with roberta')
    parser.add_argument('--base_model',
                        required=True,
                        choices=['roberta_base', 'roberta_large'])
    parser.add_argument('--from_pretrained',
                        type=str,
                        required=True,
                        help='pretrained model directory or tag')
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=512,
                        help='max sentence length, should not greater than 512')
    parser.add_argument('--batch_size', type=int, default=32, help='batchsize')
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help='data directory includes train / develop data')
    parser.add_argument('--init_checkpoint',
                        type=str,
                        default=None,
                        help='checkpoint to warm start from')
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
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help=
        "When splitting up a long document into chunks, how much stride to take between chunks."
    )
    parser.add_argument("--start_step",
                        type=int,
                        default=0,
                        help="start from which instance")
    parser.add_argument("--language",
                        type=str,
                        required=True,
                        help="language that the model based on")
    parser.add_argument(
        '--ans_path',
        type=str,
        required=True,
        help=
        "the path of the file which stores the predicted answer from last step")
    parser.add_argument(
        '--ans_idx_path',
        type=str,
        required=True,
        help=
        "the path of the file which stores the predicted answer index from last step"
    )
    parser.add_argument('--num_classes',
                        type=int,
                        required=True,
                        help="number of class")
    args = parser.parse_args()
    return args


def truncate_offset(seg, start_offset, end_offset):
    seg_len = len(seg)
    for n in range(len(start_offset) - 1, -1, -1):
        if start_offset[n] < seg_len:
            end_offset[n] = seg_len
            break
        start_offset.pop(n)
        end_offset.pop(n)


def map_fn_DuCheckList(examples, args, tokenizer):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    if args.language == 'en':
        questions = [
            examples[i]['question'].encode('ascii',
                                           errors='replace').decode('UTF-8')
            for i in range(len(examples))
        ]
        contexts = [
            examples[i]['context'].encode('ascii',
                                          errors='replace').decode('UTF-8')
            for i in range(len(examples))
        ]
    else:
        questions = [examples[i]['question'] for i in range(len(examples))]
        contexts = [examples[i]['context'] for i in range(len(examples))]
    tokenized_examples = tokenizer(questions,
                                   contexts,
                                   stride=args.doc_stride,
                                   max_seq_len=args.max_seq_len)
    tokenized_examples = convert_tokenizer_res_to_old_version(
        tokenized_examples)

    log.debug('\nexample: %d' % len(examples))
    log.debug('feature: %d\n' % len(tokenized_examples))

    # For validation, there is no need to compute start and end positions
    for i, tokenized_example in enumerate(tokenized_examples):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example['token_type_ids']

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = tokenized_example['overflow_to_sample']
        tokenized_examples[i]["example_id"] = examples[sample_index]['id']
        tokenized_examples[i]['question'] = examples[sample_index]['question']
        tokenized_examples[i]['context'] = examples[sample_index]['context']
        tokenized_examples[i]['sent_token'] = examples[sample_index][
            'sent_token']

    return tokenized_examples


def init_roberta_var(args):
    if args.language == 'ch':
        tokenizer = RobertaTokenizer.from_pretrained(args.from_pretrained)
    else:
        tokenizer = RobertaBPETokenizer.from_pretrained(args.from_pretrained)

    model = RobertaForQuestionAnswering.from_pretrained(
        args.from_pretrained, num_classes=args.num_classes)
    map_fn = partial(map_fn_DuCheckList, args=args, tokenizer=tokenizer)
    dev_ds = RCInterpret().read(args.data_dir)

    dev_ds.map(map_fn, batched=True)
    dev_batch_sampler = paddle.io.BatchSampler(dev_ds,
                                               batch_size=args.batch_size,
                                               shuffle=False)
    batchify_fn = lambda samples, fn=Dict(
        {
            "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
            "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
            "offset_mapping": Pad(axis=0, pad_val=tokenizer.pad_token_id),
            "overflow_to_sample": Stack(dtype='int32'),
        }): fn(samples)

    dev_dataloader = paddle.io.DataLoader(dataset=dev_ds,
                                          batch_sampler=dev_batch_sampler,
                                          collate_fn=batchify_fn,
                                          return_list=True)

    return model, tokenizer, dev_dataloader, dev_ds


def ch_per_example(args, scores_in_one_example, prev_context_tokens, dev_ds,
                   prev_example_idx, ans_dic, ans_idx_dic, offset, out_handle):
    total_score = scores_in_one_example[-1]
    assert len(prev_context_tokens) == len(total_score)
    token_score_dict = []
    for idx in range(len(total_score)):
        token_score_dict.append([idx, offset[idx], total_score[idx]])

    prev_example = dev_ds.data[prev_example_idx]
    char_attribution_dict = match(
        prev_example['context'] + prev_example['title'],
        prev_example['sent_token'], token_score_dict)
    result['id'] = prev_example['id']
    result['question'] = prev_example['question']
    result['title'] = prev_example['title']
    result['context'] = prev_example['context'] + prev_example['title']
    result['pred_label'] = ans_dic[str(result['id'])]
    result['pred_feature'] = ans_idx_dic[str(result['id'])]

    result['char_attri'] = collections.OrderedDict()
    for token_info in sorted(char_attribution_dict,
                             key=lambda x: x[2],
                             reverse=True):
        result['char_attri'][str(
            token_info[0])] = [str(token_info[1]),
                               float(token_info[2])]

    out_handle.write(json.dumps(result, ensure_ascii=False) + '\n')


def en_per_example(inter_score, result, ans_dic, ans_idx_dic, offset,
                   out_handle):
    sorted_token = []
    for i in range(len(inter_score)):
        sorted_token.append([i, offset[i], inter_score[i]])
    char_attribution_dict = match(result['context'], result['sent_token'],
                                  sorted_token)

    result['pred_label'] = ans_dic[str(result['id'])]
    result['pred_feature'] = ans_idx_dic[str(result['id'])]
    result['char_attri'] = collections.OrderedDict()
    for token_info in sorted(char_attribution_dict,
                             key=lambda x: x[2],
                             reverse=True):
        result['char_attri'][str(
            token_info[0])] = [str(token_info[1]),
                               float(token_info[2])]
    result.pop('sent_token')

    out_handle.write(json.dumps(result, ensure_ascii=False) + '\n')


def load_pred_data(ans_path, ans_idx_path):
    f = open(ans_path, "r")
    ans_dic = json.loads(f.read())
    f.close()
    f = open(ans_idx_path, "r")
    ans_idx_dic = json.loads(f.read())
    f.close()
    return ans_dic, ans_idx_dic


def extract_attention_scores(args, model, result, fwd_args, fwd_kwargs,
                             prev_example_idx, example_idx, prev_context_tokens,
                             scores_in_one_example, dev_ds, ans_dic,
                             ans_idx_dic, context_tokens, offset, prev_offset,
                             out_handle):
    with paddle.no_grad():
        # start_logits: (bsz, seq); end_logits: (bsz, seq); cls_logits: (bsz, 2)
        # attention: list((bsz, head, seq, seq) * 12); embedded: (bsz, seq, emb)
        _, start_logits, end_logits, cls_logits, attentions, embedded = model.forward_interpret(
            *fwd_args, **fwd_kwargs)

    # Attention score equals to the mean of attention of each token in the question
    attentions = attentions[-1][:, :, 1:SEP_idx, :].mean(2).mean(
        1)  # attentions: (bsz, seq_len)
    context_score = attentions[0, SEP_idx +
                               add_idx:-1]  # context_score: Tensor(context)
    context_norm_score = context_score / context_score.sum(-1)

    if args.language == 'ch':
        if prev_example_idx is None or prev_example_idx == example_idx:
            scores_in_one_example.append(context_norm_score.numpy().tolist())
        else:
            ch_per_example(args, scores_in_one_example, prev_context_tokens,
                           dev_ds, prev_example_idx, ans_dic, ans_idx_dic,
                           prev_offset, out_handle)
            scores_in_one_example = [context_norm_score.numpy().tolist()]
        prev_example_idx = example_idx
        prev_context_tokens = context_tokens
        prev_offset = offset
    else:
        en_per_example(context_norm_score, result, ans_dic, ans_idx_dic, offset,
                       out_handle)
    return prev_example_idx, prev_context_tokens, scores_in_one_example, prev_offset


def extract_integrated_gradient_scores(
        args, dev_ds, model, result, fwd_args, fwd_kwargs, SEP_idx, add_idx,
        prev_example_idx, example_idx, scores_in_one_example,
        prev_context_tokens, ans_dic, ans_idx_dic, context_tokens, offset,
        prev_offset, out_handle):
    embedded_grads_list = []  # [Tensor(1, seq_len, embed_size)]
    with open(os.path.join(args.output_dir, 'predict_feature_index'),
              'r') as f_feature_index:
        feature_index_dict = json.load(f_feature_index)
    example = dev_ds.data[example_idx]
    example_id = example['id']
    start_index, end_index = feature_index_dict[str(example_id)]

    for i in range(args.n_samples):
        # embedded_start_grad
        # start_logits: (bsz, seq); embedded: (bsz, seq, emb)
        _, start_logits, _, _, _, embedded = model.forward_interpret(
            *fwd_args,
            **fwd_kwargs,
            noise='integrated',
            i=i,
            n_samples=args.n_samples)

        start_logit = start_logits[:, start_index].sum()
        start_logit.backward(retain_graph=False)
        embedded_start_grad = embedded.grad
        model.clear_gradients()
        # embedded_end_grad
        # end_logits: (bsz, seq); embedded: (bsz, seq, emb)
        _, _, end_logits, _, _, embedded = model.forward_interpret(
            *fwd_args,
            **fwd_kwargs,
            noise='integrated',
            i=i,
            n_samples=args.n_samples)
        end_logit = end_logits[:, end_index].sum()
        end_logit.backward(retain_graph=False)
        embedded_end_grad = embedded.grad
        model.clear_gradients()

        embedded_grad = (embedded_start_grad + embedded_end_grad) / 2
        embedded_grads_list.append(embedded_grad)

        if i == 0:
            baseline_embedded = embedded  # Tensor(1, seq_len, embed_size)
        elif i == args.n_samples - 1:
            pred_embedded = embedded  # Tensor(1, seq_len, embed_size)

    embedded_grads_tensor = paddle.to_tensor(embedded_grads_list,
                                             dtype='float32',
                                             place=paddle.CUDAPlace(0),
                                             stop_gradient=True)

    trapezoidal_grads = (embedded_grads_tensor[1:] + embedded_grads_tensor[:-1]
                         ) / 2  # Tensor(n_samples-1, 1, seq_len, embed_size)
    integral_grads = trapezoidal_grads.sum(0) / trapezoidal_grads.shape[
        0]  # Tensor(1, seq_len, embed_size)xw

    inter_score = (pred_embedded - baseline_embedded
                   ) * integral_grads  # Tensor(1, seq_len, embed_size)
    inter_score = inter_score.sum(-1)  # Tensor(1, seq_len)
    inter_score.stop_gradient = True

    context_score = inter_score[0, SEP_idx + add_idx:-1]
    context_norm_score = context_score / context_score.sum(-1)
    if args.language == 'ch':
        if prev_example_idx is None or prev_example_idx == example_idx:
            scores_in_one_example.append(context_norm_score.numpy().tolist())
        else:
            ch_per_example(args, scores_in_one_example, prev_context_tokens,
                           dev_ds, prev_example_idx, ans_dic, ans_idx_dic,
                           prev_offset, out_handle)
            scores_in_one_example = [context_norm_score.numpy().tolist()]
        prev_example_idx = example_idx
        prev_context_tokens = context_tokens
        prev_offset = offset
    else:
        en_per_example(context_norm_score, result, ans_dic, ans_idx_dic, offset,
                       out_handle)
    return prev_example_idx, prev_context_tokens, scores_in_one_example, prev_offset


if __name__ == "__main__":
    args = get_args()
    if args.language == 'ch':
        add_idx = 1
    else:
        add_idx = 2

    ans_dic, ans_idx_dic = load_pred_data(args.ans_path, args.ans_idx_path)
    if args.base_model.startswith('roberta'):
        model, tokenizer, dataloader, dev_ds = init_roberta_var(args)
    else:
        raise ValueError('unsupported base model name.')

    with paddle.amp.auto_cast(enable=args.use_amp), \
        open(os.path.join(args.output_dir, 'interpret' + f'.{args.inter_mode}'), 'w') as out_handle:

        sd = paddle.load(args.init_checkpoint)
        model.set_dict(sd)
        log.debug('load model from %s' % args.init_checkpoint)

        err_total = []
        lime_score_total = []
        lime_relative_err_total = []
        lime_err_total = []

        # Second forward: evidence extraction
        scores_in_one_example = []
        prev_example_idx = None
        prev_context_tokens = None
        prev_offset = None

        get_subword_ids = lambda word: map(
            str, tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)))
        for step, d in tqdm(enumerate(dataloader)):
            if step < args.start_step:
                continue

            model.train()

            result = {}
            input_ids, segment_ids, offset_map, example_idx = d
            fwd_args = [input_ids, segment_ids]
            fwd_kwargs = {}

            SEP_idx = input_ids.numpy()[0].tolist().index(
                tokenizer.sep_token_id)
            context_ids = input_ids[0, SEP_idx + add_idx:-1]
            offset = offset_map[0, SEP_idx + add_idx:-1]
            context_tokens = tokenizer.convert_ids_to_tokens(
                context_ids.numpy().tolist())

            if args.language == 'en':
                example = dev_ds.data[step]
                result['id'] = example['id']
                result['question'] = example['question']
                result['title'] = example['title']
                result['context'] = example['context'] + example['title']
                result['sent_token'] = example['sent_token']

            if args.inter_mode == "attention":
                prev_example_idx, prev_context_tokens, scores_in_one_example, prev_offset = extract_attention_scores(
                    args, model, result, fwd_args, fwd_kwargs, prev_example_idx,
                    example_idx, prev_context_tokens, scores_in_one_example,
                    dev_ds, ans_dic, ans_idx_dic, context_tokens, offset,
                    prev_offset, out_handle)

            elif args.inter_mode == 'integrated_gradient':
                prev_example_idx, prev_context_tokens, scores_in_one_example, prev_offset = extract_integrated_gradient_scores(
                    args, dev_ds, model, result, fwd_args, fwd_kwargs, SEP_idx,
                    add_idx, prev_example_idx, example_idx,
                    scores_in_one_example, prev_context_tokens, ans_dic,
                    ans_idx_dic, context_tokens, offset, prev_offset,
                    out_handle)
            else:
                raise KeyError(f"Unkonwn interpretable mode: {args.inter_mode}")

        # Deal with last example
        if args.language == 'ch':

            feature = dev_ds.new_data[-1]
            input_ids = feature['input_ids']
            SEP_idx = input_ids.index(tokenizer.sep_token_id)
            context_ids = input_ids[SEP_idx + 1:-1]
            offset = feature['offset_mapping'][SEP_idx + 1:-1]
            context_tokens = tokenizer.convert_ids_to_tokens(context_ids)

            ch_per_example(args, scores_in_one_example, context_tokens, dev_ds,
                           -1, ans_dic, ans_idx_dic, offset, out_handle)
