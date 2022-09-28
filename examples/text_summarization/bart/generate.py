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
import sys
import argparse
import random
import time
from functools import partial
from pprint import pprint
import numpy as np
import paddle
from paddle.io import BatchSampler, DataLoader
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Tuple, Stack
from paddlenlp.transformers import BartForConditionalGeneration, BartTokenizer
from utils import convert_example, compute_metrics

summarization_name_mapping = {"cnn_dailymail": ("article", "highlights")}


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--model_name_or_path",
                        default="bart-base",
                        type=str,
                        required=True,
                        help="Path to pre-trained model. ")
    parser.add_argument(
        "--dataset_name",
        default="cnn_dailymail",
        type=str,
        required=True,
        help="The name of the dataset to use. Selected in the list: " +
        ", ".join(summarization_name_mapping.keys()))
    parser.add_argument(
        '--output_path',
        type=str,
        default='generate.txt',
        help='The file path where the infer result will be saved.')
    parser.add_argument(
        "--max_source_length",
        default=1024,
        type=int,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--min_target_length",
        default=0,
        type=int,
        help=
        "The minimum total sequence length for target text when generating. ")
    parser.add_argument(
        "--max_target_length",
        default=142,
        type=int,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument('--decode_strategy',
                        default='greedy_search',
                        type=str,
                        help='The decode strategy in generation.')
    parser.add_argument(
        '--top_k',
        default=2,
        type=int,
        help=
        'The number of highest probability vocabulary tokens to keep for top-k sampling.'
    )
    parser.add_argument('--top_p',
                        default=1.0,
                        type=float,
                        help='The cumulative probability for top-p sampling.')
    parser.add_argument('--num_beams',
                        default=1,
                        type=int,
                        help='The number of beams for beam search.')
    parser.add_argument(
        '--length_penalty',
        default=0.6,
        type=float,
        help='The exponential penalty to the sequence length for beam search.')
    parser.add_argument(
        '--early_stopping',
        default=False,
        type=eval,
        help=
        'Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.'
    )
    parser.add_argument("--diversity_rate",
                        default=0.0,
                        type=float,
                        help="The diversity of beam search. ")
    parser.add_argument(
        '--faster',
        action='store_true',
        help='Whether to process inference using faster transformer. ')
    parser.add_argument(
        '--use_fp16_decoding',
        action='store_true',
        help=
        'Whether to use fp16 when using faster transformer. Only works when using faster transformer. '
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU/CPU for testing or evaluation.")
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="random seed for initialization")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu", "xpu"],
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        default=True,
        type=bool,
        help="Whether to ignore the tokens corresponding to "
        "padded labels in the loss computation or not.",
    )
    parser.add_argument("--logging_steps",
                        type=int,
                        default=100,
                        help="Log every X updates steps.")
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
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    model = BartForConditionalGeneration.from_pretrained(
        args.model_name_or_path)
    dataset = load_dataset(args.dataset_name, splits=["dev"])
    trans_func = partial(
        convert_example,
        text_column=summarization_name_mapping[args.dataset_name][0],
        summary_column=summarization_name_mapping[args.dataset_name][1],
        tokenizer=tokenizer,
        decoder_start_token_id=model.bart.decoder_start_token_id,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
        is_train=False)
    batchify_fn = lambda samples, fn=Tuple(
        Stack(dtype="int64"),  # input_ids
        Stack(dtype="int64"),  # attention mask
        Stack(dtype="int32"),  # mem_seq_lens
        Stack(dtype="int64"),  # decoder_input_ids
        Stack(dtype="int64"),  # labels
    ): fn(samples)

    dataset = dataset.map(trans_func, lazy=True)
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
                                  seq_lens=mem_seq_lens,
                                  max_length=args.max_target_length,
                                  min_length=args.min_target_length,
                                  decode_strategy=args.decode_strategy,
                                  top_k=args.top_k,
                                  top_p=args.top_p,
                                  num_beams=args.num_beams,
                                  length_penalty=args.length_penalty,
                                  early_stopping=args.early_stopping,
                                  diversity_rate=args.diversity_rate,
                                  use_faster=args.faster)
        total_time += (time.time() - start_time)
        if step % args.logging_steps == 0:
            print('step %d - %.3fs/step' %
                  (step, total_time / args.logging_steps))
            total_time = 0.0
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
        start_time = time.time()

    rouge_result, decoded_preds = compute_metrics(
        all_preds, all_labels, tokenizer, args.ignore_pad_token_for_loss)
    print("Rouge result: ", rouge_result)
    with open(args.output_path, 'w', encoding='utf-8') as fout:
        for decoded_pred in decoded_preds:
            fout.write(decoded_pred + '\n')
    print('Save generated result into: %s' % args.output_path)


if __name__ == '__main__':
    args = parse_args()
    pprint(args)
    generate(args)
