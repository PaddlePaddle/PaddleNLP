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
import time
import logging
import json
import collections
from functools import partial
from pathlib import Path
import argparse

import paddle
from paddle.io import DataLoader
from paddlenlp.data import Pad, Dict
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics.squad import squad_evaluate

from squad import RCInterpret
from squad import compute_prediction
from paddlenlp.transformers.roberta.tokenizer import RobertaTokenizer, RobertaBPETokenizer

from roberta.modeling import RobertaForQuestionAnswering

sys.path.append('../../..')
from model_interpretation.utils import convert_tokenizer_res_to_old_version

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
                        default=128,
                        help='max sentence length, should not greater than 512')
    parser.add_argument('--batch_size', type=int, default=32, help='batchsize')
    parser.add_argument('--epoch', type=int, default=3, help='epoch')
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
    parser.add_argument("--language",
                        type=str,
                        required=True,
                        help="language that the model based on")
    args = parser.parse_args()
    return args


def map_fn_DuCheckList(examples, args, tokenizer):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    #NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
    # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
    if args.language == 'en':
        contexts = [
            examples[i]['context'].encode('ascii',
                                          errors='replace').decode('UTF-8')
            for i in range(len(examples))
        ]
        questions = [
            examples[i]['question'].encode('ascii',
                                           errors='replace').decode('UTF-8')
            for i in range(len(examples))
        ]
    else:
        contexts = [examples[i]['context'] for i in range(len(examples))]
        questions = [examples[i]['question'] for i in range(len(examples))]

    tokenized_examples = tokenizer(questions,
                                   contexts,
                                   stride=args.doc_stride,
                                   max_seq_len=args.max_seq_len)
    tokenized_examples = convert_tokenizer_res_to_old_version(
        tokenized_examples)

    # For validation, there is no need to compute start and end positions
    for i, tokenized_example in enumerate(tokenized_examples):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example['token_type_ids']

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = tokenized_example['overflow_to_sample']
        tokenized_examples[i]["example_id"] = examples[sample_index]['id']

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        if args.language == 'ch':
            tokenized_examples[i]["offset_mapping"] = [
                (o if sequence_ids[k] == 1 else None)
                for k, o in enumerate(tokenized_example["offset_mapping"])
            ]
        else:
            n = tokenized_example['offset_mapping'].index(
                (0, 0), 1) + 2  # context start position
            m = len(tokenized_example['offset_mapping']
                    ) - 1  # context end position + 1
            tokenized_examples[i]["offset_mapping"] = [
                (o if n <= k <= m else None)
                for k, o in enumerate(tokenized_example["offset_mapping"])
            ]
    return tokenized_examples


def init_roberta_var(args):
    if args.language == 'ch':
        tokenizer = RobertaTokenizer.from_pretrained(args.from_pretrained)
    else:
        tokenizer = RobertaBPETokenizer.from_pretrained(args.from_pretrained)

    model = RobertaForQuestionAnswering.from_pretrained(args.from_pretrained)
    map_fn = partial(map_fn_DuCheckList, args=args, tokenizer=tokenizer)
    dev_ds = RCInterpret().read(args.data_dir)
    dev_ds.map(map_fn, batched=True)
    dev_batch_sampler = paddle.io.BatchSampler(dev_ds,
                                               batch_size=args.batch_size,
                                               shuffle=False)
    batchify_fn = lambda samples, fn=Dict(
        {
            "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
            "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id)
        }): fn(samples)

    dev_dataloader = paddle.io.DataLoader(dataset=dev_ds,
                                          batch_sampler=dev_batch_sampler,
                                          collate_fn=batchify_fn,
                                          return_list=True)

    return model, tokenizer, dev_dataloader, dev_ds


@paddle.no_grad()
def evaluate(model, data_loader, args):
    model.eval()

    all_start_logits = []
    all_end_logits = []
    all_cls_logits = []
    tic_eval = time.time()

    for batch in data_loader:
        input_ids, token_type_ids = batch
        loss, start_logits_tensor, end_logits_tensor, cls_logits = model(
            input_ids, token_type_ids)
        for idx in range(start_logits_tensor.shape[0]):
            if len(all_start_logits) % 1000 == 0 and len(all_start_logits):
                log.debug("Processing example: %d" % len(all_start_logits))
                log.debug('time per 1000:%.1f' % (time.time() - tic_eval))
                tic_eval = time.time()

            all_start_logits.append(start_logits_tensor.numpy()[idx])
            all_end_logits.append(end_logits_tensor.numpy()[idx])

    all_predictions, all_nbest_json, scores_diff_json, all_feature_index = compute_prediction(
        data_loader.dataset.data, data_loader.dataset.new_data,
        (all_start_logits, all_end_logits), True, 20, args.max_seq_len, 0.0)

    # Can also write all_nbest_json and scores_diff_json files if needed
    with open(os.path.join(args.output_dir, 'predict_ans'), 'w') as f_ans_pred:
        f_ans_pred.write(
            json.dumps(all_predictions, ensure_ascii=False, indent=4) + "\n")
    with open(os.path.join(args.output_dir, 'predict_feature_index'),
              'w') as f_feature_index:
        f_feature_index.write(
            json.dumps(all_feature_index, ensure_ascii=False, indent=4) + "\n")

    #squad_evaluate(examples=data_loader.dataset.data, preds=all_predictions, na_probs=scores_diff_json)
    #model.train()


if __name__ == "__main__":
    args = get_args()
    if args.base_model.startswith('roberta'):
        model, tokenizer, dataloader, dev_ds = init_roberta_var(args)
    else:
        raise ValueError('unsupported base model name.')

    with paddle.amp.auto_cast(enable=args.use_amp):
        sd = paddle.load(args.init_checkpoint)
        model.set_dict(sd)
        log.debug('load model from %s' % args.init_checkpoint)
        evaluate(model, dataloader, args)
