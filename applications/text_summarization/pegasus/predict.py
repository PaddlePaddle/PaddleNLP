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
from datasets import load_dataset
from paddle.io import BatchSampler, DataLoader
from utils import compute_metrics, convert_example

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.transformers import (
    PegasusChineseTokenizer,
    PegasusForConditionalGeneration,
)


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--init_checkpoint_dir",
        default="IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese",
        type=str,
        required=True,
        help="Path to pre-trained model. ",
    )
    parser.add_argument(
        "--prefict_file", type=str, required=False, default="data/valid.json", help="Predict data path."
    )
    parser.add_argument(
        "--output_path", type=str, default="generate.txt", help="The file path where the infer result will be saved."
    )
    parser.add_argument(
        "--max_source_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--min_target_length",
        default=0,
        type=int,
        help="The minimum total sequence length for target text when generating. ",
    )
    parser.add_argument(
        "--max_target_length",
        default=64,
        type=int,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--decode_strategy", default="greedy_search", type=str, help="The decode strategy in generation."
    )
    parser.add_argument(
        "--top_k",
        default=2,
        type=int,
        help="The number of highest probability vocabulary tokens to keep for top-k sampling.",
    )
    parser.add_argument("--top_p", default=1.0, type=float, help="The cumulative probability for top-p sampling.")
    parser.add_argument("--num_beams", default=1, type=int, help="The number of beams for beam search.")
    parser.add_argument(
        "--length_penalty",
        default=0.6,
        type=float,
        help="The exponential penalty to the sequence length for beam search.",
    )
    parser.add_argument(
        "--early_stopping",
        default=False,
        type=eval,
        help="Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.",
    )
    parser.add_argument("--diversity_rate", default=0.0, type=float, help="The diversity of beam search. ")
    parser.add_argument(
        "--faster", action="store_true", help="Whether to process inference using faster transformer. "
    )
    parser.add_argument(
        "--use_fp16_decoding",
        action="store_true",
        help="Whether to use fp16 when using faster transformer. Only works when using faster transformer. ",
    )
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size per GPU/CPU for testing or evaluation.")
    parser.add_argument("--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu", "xpu"],
        help="The device to select to train the model, is must be cpu/gpu/xpu.",
    )
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
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
    tokenizer = PegasusChineseTokenizer.from_pretrained(args.init_checkpoint_dir)
    model = PegasusForConditionalGeneration.from_pretrained(args.init_checkpoint_dir)
    dataset = load_dataset("json", data_files=args.prefict_file, split="train")
    remove_columns = ["content", "title"]
    trans_func = partial(
        convert_example,
        text_column="content",
        summary_column="title",
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    dataset = dataset.map(trans_func, batched=True, load_from_cache_file=True, remove_columns=remove_columns)
    batch_sampler = BatchSampler(dataset, batch_size=args.batch_size, shuffle=False)
    batchify_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    data_loader = DataLoader(
        dataset=dataset, batch_sampler=batch_sampler, num_workers=0, collate_fn=batchify_fn, return_list=True
    )
    data_loader.pin_memory = False

    model.eval()
    total_time = 0.0
    start_time = time.time()
    all_preds = []
    all_labels = []
    for step, batch in enumerate(data_loader):
        labels = batch.pop("labels").numpy()
        preds, _ = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=args.max_target_length,
            min_length=args.min_target_length,
            decode_strategy=args.decode_strategy,
            top_k=args.top_k,
            top_p=args.top_p,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            early_stopping=args.early_stopping,
            diversity_rate=args.diversity_rate,
            use_fast=args.faster,
        )
        total_time += time.time() - start_time
        if step % args.logging_steps == 0:
            print("step %d - %.3fs/step" % (step, total_time / args.logging_steps))
            total_time = 0.0
        all_preds.extend(
            tokenizer.batch_decode(preds.numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        )
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        all_labels.extend(tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False))
        start_time = time.time()

    compute_metrics(all_preds, all_labels)
    with open(args.output_path, "w", encoding="utf-8") as fout:
        for decoded_pred in all_preds:
            fout.write(decoded_pred + "\n")
    print("Save generated result into: %s" % args.output_path)


if __name__ == "__main__":
    args = parse_args()
    pprint(args)
    generate(args)
