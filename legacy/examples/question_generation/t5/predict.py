# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import argparse
import random
import time
from functools import partial
from pprint import pprint

import numpy as np
import paddle
from paddle.io import BatchSampler, DataLoader
from utils import compute_metrics, convert_example

from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import T5ForConditionalGeneration, T5Tokenizer


# yapf: disable
def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--model_name_or_path", default="t5-base", type=str, required=True, help="Path to pre-trained model. ")
    parser.add_argument("--dataset_name", default="squad", type=str, required=True, help="The name of the dataset to use. Selected in the list: " + "squad")
    parser.add_argument('--output_path', type=str, default='generate.txt', help='The file path where the infer result will be saved.')
    parser.add_argument("--max_source_length", default=1024, type=int, help="The maximum total input sequence length after tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--min_target_length", default=0, type=int, help="The minimum total sequence length for target text when generating. ")
    parser.add_argument("--max_target_length", default=142, type=int, help="The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded during ``evaluate`` and ``predict``.",)
    parser.add_argument('--decode_strategy', default='greedy_search', type=str, help='The decode strategy in generation.')
    parser.add_argument('--top_k', default=2, type=int, help='The number of highest probability vocabulary tokens to keep for top-k sampling.')
    parser.add_argument('--top_p', default=1.0, type=float, help='The cumulative probability for top-p sampling.')
    parser.add_argument('--num_beams', default=1, type=int, help='The number of beams for beam search.')
    parser.add_argument('--length_penalty', default=0.6, type=float, help='The exponential penalty to the sequence length for beam search.')
    parser.add_argument('--early_stopping', default=False, type=eval, help='Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.')
    parser.add_argument("--diversity_rate", default=0.0, type=float, help="The diversity of beam search. ")
    parser.add_argument('--faster', action='store_true', help='Whether to process inference using FastGeneration. ')
    parser.add_argument('--use_fp16_decoding', action='store_true', help='Whether to use fp16 when using FastGeneration. Only works when using FastGeneration. ')
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU/CPU for testing or evaluation.")
    parser.add_argument("--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu", "xpu"], help="The device to select to train the model, is must be cpu/gpu/xpu.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--is_debug", action='store_true', help="Whether to debug.")
    parser.add_argument("--ignore_pad_token_for_loss", action='store_true', help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.")

    args = parser.parse_args()
    return args


def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


@paddle.no_grad()
def generate(args):
    paddle.set_device(args.device)
    set_seed(args)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    dataset = load_dataset(args.dataset_name, splits=["dev_v1"])
    # dataset = load_dataset(args.dataset_name, splits=["dev_v2"])
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        decoder_start_token_id=model.t5.bos_token_id,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
        is_train=False)

    def batchify_fn(samples, tokenizer):
        fn = Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # input_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"
                ),  # attention_mask
            Pad(axis=0, pad_val=-100, dtype="int64"),  # mem_seq_lens
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"
                ),  # decoder_input_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # labels
        )
        return fn(samples)

    dataset = dataset.map(trans_func, lazy=True)

    # debug
    if args.is_debug:
        dataset.data = dataset.data[:20]
        dataset.new_data = dataset.new_data[:20]

    batch_sampler = BatchSampler(dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)
    data_loader = DataLoader(dataset=dataset,
                             batch_sampler=batch_sampler,
                             num_workers=0,
                             collate_fn=batchify_fn,
                             return_list=True)
    data_loader.pin_memory = False

    model.eval()
    total_time = 0.0
    start_time = time.time()
    all_preds = []
    all_labels = []
    for step, batch in enumerate(data_loader):
        input_ids, _, mem_seq_lens, _, labels = batch
        preds, _ = model.generate(input_ids=input_ids,
                                  max_length=args.max_target_length,
                                  min_length=args.min_target_length,
                                  decode_strategy=args.decode_strategy,
                                  top_k=args.top_k,
                                  top_p=args.top_p,
                                  num_beams=args.num_beams,
                                  length_penalty=args.length_penalty,
                                  early_stopping=args.early_stopping,
                                  diversity_rate=args.diversity_rate,
                                  use_fast=args.faster)
        total_time += (time.time() - start_time)
        if step % args.logging_steps == 0:
            print('step %d - %.3fs/step' %
                  (step, total_time / args.logging_steps))
            total_time = 0.0
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
        start_time = time.time()

    bleu_result, decoded_preds, decoded_labels = compute_metrics(
        all_preds, all_labels, tokenizer, args.ignore_pad_token_for_loss)
    print("BLEU result: ", bleu_result)
    with open(args.output_path, 'w', encoding='utf-8') as fout:
        for decoded_pred in decoded_preds:
            fout.write(' '.join(decoded_pred) + '\n')
    print('Save generated result into: %s' % args.output_path)
    with open(args.output_path + '.reference.txt', 'w',
              encoding='utf-8') as fout:
        for decoded_label in decoded_labels:
            fout.write(' '.join(decoded_label) + '\n')
    print('Save referenced labels into: {}.reference.txt'.format(args.output_path))


if __name__ == '__main__':
    args = parse_args()
    pprint(args)
    generate(args)
