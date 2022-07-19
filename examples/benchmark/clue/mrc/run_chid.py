# coding: utf-8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The HuggingFace Inc. team.
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
import time
import argparse
import json
import distutils.util
import random
from functools import partial
import contextlib

import numpy as np

import paddle
import paddle.nn as nn

from datasets import load_dataset
from paddlenlp.data import Pad, Stack, Tuple, Dict
from paddlenlp.transformers import AutoModelForMultipleChoice, AutoTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.log import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
    parser.add_argument("--model_name_or_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to pre-trained model or shortcut name.")
    parser.add_argument("--output_dir",
                        default="best_chid_model",
                        type=str,
                        help="The path of the checkpoints .")
    parser.add_argument("--save_best_model",
                        default=True,
                        type=distutils.util.strtobool,
                        help="Whether to save best model.")
    parser.add_argument("--overwrite_cache",
                        default=False,
                        type=distutils.util.strtobool,
                        help="Whether to overwrite cache for dataset.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--num_proc",
        default=None,
        type=int,
        help=
        "Max number of processes when generating cache. Already cached shards are loaded sequentially."
    )
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs."
    )
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Linear warmup proportion over total steps.")
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help=
        "Linear warmup over warmup_steps. If > 0: Override warmup_proportion")
    parser.add_argument("--adam_epsilon",
                        default=1e-6,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="random seed for initialization")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="The max value of grad norm.")
    parser.add_argument("--batch_size",
                        default=4,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size",
                        default=24,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumualte before performing a backward/update pass."
    )
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to train.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to predict.")
    parser.add_argument(
        "--max_seq_length",
        default=64,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.")
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


def calc_global_pred_results(logits):
    logits = np.array(logits)
    # [num_choices, tag_size]
    logits = np.transpose(logits)
    tmp = []
    for i, row in enumerate(logits):
        for j, col in enumerate(row):
            tmp.append((i, j, col))
    else:
        choice = set(range(i + 1))
        blanks = set(range(j + 1))
    tmp = sorted(tmp, key=lambda x: x[2], reverse=True)
    results = []
    for i, j, v in tmp:
        if (j in blanks) and (i in choice):
            results.append((i, j))
            blanks.remove(j)
            choice.remove(i)
    results = sorted(results, key=lambda x: x[1], reverse=False)
    results = [i for i, j in results]
    return results


@paddle.no_grad()
def evaluate(model, data_loader, do_predict=False):
    model.eval()
    right_num, total_num = 0, 0
    all_results = []
    for step, batch in enumerate(data_loader):
        if do_predict:
            input_ids, segment_ids, example_ids = batch
        else:
            input_ids, segment_ids, labels, example_ids = batch
        logits = model(input_ids=input_ids, token_type_ids=segment_ids)
        batch_num = example_ids.shape[0]
        l = 0
        r = batch_num - 1
        batch_results = []
        for i in range(batch_num - 1):
            if example_ids[i] != example_ids[i + 1]:
                r = i
                batch_results.extend(
                    calc_global_pred_results(logits[l:r + 1, :]))
                l = i + 1
        if l <= batch_num - 1:
            batch_results.extend(
                calc_global_pred_results(logits[l:batch_num, :]))
        if do_predict:
            all_results.extend(batch_results)
        else:
            right_num += np.sum(np.array(batch_results) == labels.numpy())
            total_num += labels.shape[0]
    model.train()
    if not do_predict:
        acc = right_num / total_num
        return acc
    return all_results


@contextlib.contextmanager
def main_process_first(desc="work"):
    if paddle.distributed.get_world_size() > 1:
        rank = paddle.distributed.get_rank()
        is_main_process = rank == 0
        main_process_desc = "main local process"
        try:
            if not is_main_process:
                # tell all replicas to wait
                logger.debug(
                    f"{rank}: waiting for the {main_process_desc} to perform {desc}"
                )
                paddle.distributed.barrier()
            yield
        finally:
            if is_main_process:
                # the wait is over
                logger.debug(
                    f"{rank}: {main_process_desc} completed {desc}, releasing all replicas"
                )
                paddle.distributed.barrier()
    else:
        yield


def run(args):
    if args.do_train:
        assert args.batch_size % args.gradient_accumulation_steps == 0, \
            "Please make sure argmument `batch_size` must be divisible by `gradient_accumulation_steps`."
    paddle.set_device(args.device)
    set_seed(args)

    max_seq_length = args.max_seq_length
    max_num_choices = 10

    def preprocess_function(examples, do_predict=False):
        SPIECE_UNDERLINE = '▁'

        def _is_chinese_char(cp):
            if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
                return True

            return False

        def is_fuhao(c):
            if c == '。' or c == '，' or c == '！' or c == '？' or c == '；' or c == '、' or c == '：' or c == '（' or c == '）' \
                    or c == '－' or c == '~' or c == '「' or c == '《' or c == '》' or c == ',' or c == '」' or c == '"' or c == '“' or c == '”' \
                    or c == '$' or c == '『' or c == '』' or c == '—' or c == ';' or c == '。' or c == '(' or c == ')' or c == '-' or c == '～' or c == '。' \
                    or c == '‘' or c == '’':
                return True
            return False

        def _tokenize_chinese_chars(text):
            """Adds whitespace around any CJK character."""
            output = []
            is_blank = False
            for index, char in enumerate(text):
                cp = ord(char)
                if is_blank:
                    output.append(char)
                    if context[index - 12:index + 1].startswith("#idiom"):
                        is_blank = False
                        output.append(SPIECE_UNDERLINE)
                else:
                    if text[index:index + 6] == "#idiom":
                        is_blank = True
                        if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
                            output.append(SPIECE_UNDERLINE)
                        output.append(char)
                    elif _is_chinese_char(cp) or is_fuhao(char):
                        if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
                            output.append(SPIECE_UNDERLINE)
                        output.append(char)
                        output.append(SPIECE_UNDERLINE)
                    else:
                        output.append(char)
            return "".join(output)

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(
                    c) == 0x202F or c == SPIECE_UNDERLINE:
                return True
            return False

        def add_tokens_for_around(tokens, pos, num_tokens):
            num_l = num_tokens // 2
            num_r = num_tokens - num_l

            if pos >= num_l and (len(tokens) - 1 - pos) >= num_r:
                tokens_l = tokens[pos - num_l:pos]
                tokens_r = tokens[pos + 1:pos + 1 + num_r]
            elif pos <= num_l:
                tokens_l = tokens[:pos]
                right_len = num_tokens - len(tokens_l)
                tokens_r = tokens[pos + 1:pos + 1 + right_len]
            elif (len(tokens) - 1 - pos) <= num_r:
                tokens_r = tokens[pos + 1:]
                left_len = num_tokens - len(tokens_r)
                tokens_l = tokens[pos - left_len:pos]
            else:
                raise ValueError('impossible')

            return tokens_l, tokens_r

        max_tokens_for_doc = max_seq_length - 3
        num_tokens = max_tokens_for_doc - 5
        num_examples = len(examples.data["candidates"])
        if do_predict:
            result = {"input_ids": [], "token_type_ids": [], "example_ids": []}
        else:
            result = {
                "input_ids": [],
                "token_type_ids": [],
                "labels": [],
                "example_ids": []
            }
        for idx in range(num_examples):
            candidate = 0
            options = examples.data['candidates'][idx]

            # Each content may have several sentences.
            for context in examples.data['content'][idx]:
                context = context.replace("“", "\"").replace("”", "\"").replace("——", "--"). \
                    replace("—", "-").replace("―", "-").replace("…", "...").replace("‘", "\'").replace("’", "\'")
                context = _tokenize_chinese_chars(context)
                paragraph_text = context.strip()
                doc_tokens = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                all_doc_tokens = []
                for (i, token) in enumerate(doc_tokens):
                    if '#idiom' in token:
                        sub_tokens = [str(token)]
                    else:
                        sub_tokens = tokenizer.tokenize(token)
                    for sub_token in sub_tokens:
                        all_doc_tokens.append(sub_token)
                tags = [blank for blank in doc_tokens if '#idiom' in blank]

                # Each sentence may have several tags
                for tag_index, tag in enumerate(tags):
                    pos = all_doc_tokens.index(tag)

                    tmp_l, tmp_r = add_tokens_for_around(
                        all_doc_tokens, pos, num_tokens)
                    num_l = len(tmp_l)
                    num_r = len(tmp_r)
                    tokens_l = []
                    for token in tmp_l:
                        if '#idiom' in token and token != tag:
                            # Mask tag which is not considered in this new sample.
                            # Each idiom has four words, so 4 mask tokens are used.
                            tokens_l.extend(['[MASK]'] * 4)
                        else:
                            tokens_l.append(token)
                    tokens_l = tokens_l[-num_l:]
                    del tmp_l

                    tokens_r = []
                    for token in tmp_r:
                        if '#idiom' in token and token != tag:
                            tokens_r.extend(['[MASK]'] * 4)
                        else:
                            tokens_r.append(token)
                    tokens_r = tokens_r[:num_r]
                    del tmp_r

                    tokens_list = []
                    # Each tag has ten choices, and the shape of each new
                    # example is [num_choices, seq_len]
                    for i, elem in enumerate(options):
                        option = tokenizer.tokenize(elem)
                        tokens = option + ['[SEP]'] + tokens_l + ['[unused1]'
                                                                  ] + tokens_r
                        tokens_list.append(tokens)
                    new_data = tokenizer(tokens_list, is_split_into_words=True)
                    # Final shape of input_ids: [batch_size, num_choices, seq_len]
                    result["input_ids"].append(new_data["input_ids"])
                    result["token_type_ids"].append(new_data["token_type_ids"])
                    result["example_ids"].append(idx)
                    if not do_predict:
                        label = examples.data["answers"][idx]["candidate_id"][
                            candidate]
                        result["labels"].append(label)
                    candidate += 1
            if (idx + 1) % 10000 == 0:
                logger.info("%d samples have been processed." % (idx + 1))
        return result

    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    model = AutoModelForMultipleChoice.from_pretrained(
        args.model_name_or_path, num_choices=max_num_choices)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    train_ds, dev_ds, test_ds = load_dataset(
        "clue", "chid", split=["train", "validation", "test"])

    if args.do_train:
        args.batch_size = int(args.batch_size /
                              args.gradient_accumulation_steps)
        column_names = train_ds.column_names
        with main_process_first(desc="train dataset map pre-processing"):
            train_ds = train_ds.map(
                partial(preprocess_function),
                batched=True,
                batch_size=len(train_ds),
                num_proc=args.num_proc,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on train dataset")
        batchify_fn = lambda samples, fn=Dict(
            {
                'input_ids': Pad(axis=1, pad_val=tokenizer.pad_token_id
                                 ),  # input
                'token_type_ids': Pad(
                    axis=1, pad_val=tokenizer.pad_token_type_id),  # segment
                'labels': Stack(dtype="int64"),  # label
                'example_ids': Stack(dtype="int64"),  # example id
            }): fn(samples)

        train_batch_sampler = paddle.io.DistributedBatchSampler(
            train_ds, batch_size=args.batch_size, shuffle=True)
        train_data_loader = paddle.io.DataLoader(
            dataset=train_ds,
            batch_sampler=train_batch_sampler,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)
        with main_process_first(desc="evaluate dataset map pre-processing"):
            dev_ds = dev_ds.map(partial(preprocess_function),
                                batched=True,
                                batch_size=len(dev_ds),
                                remove_columns=column_names,
                                num_proc=args.num_proc,
                                load_from_cache_file=args.overwrite_cache,
                                desc="Running tokenizer on validation dataset")

        dev_batch_sampler = paddle.io.BatchSampler(
            dev_ds, batch_size=args.eval_batch_size, shuffle=False)

        dev_data_loader = paddle.io.DataLoader(dataset=dev_ds,
                                               batch_sampler=dev_batch_sampler,
                                               collate_fn=batchify_fn,
                                               return_list=True)

        num_training_steps = int(
            args.max_steps /
            args.gradient_accumulation_steps) if args.max_steps >= 0 else int(
                len(train_data_loader) * args.num_train_epochs /
                args.gradient_accumulation_steps)

        warmup = args.warmup_steps if args.warmup_steps > 0 else args.warmup_proportion
        lr_scheduler = LinearDecayWithWarmup(args.learning_rate,
                                             num_training_steps, warmup)
        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        decay_params = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
        grad_clip = paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm)
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params,
            grad_clip=grad_clip)

        loss_fct = nn.CrossEntropyLoss()

        model.train()
        global_step = 0
        best_acc = 0.0
        tic_train = time.time()
        for epoch in range(args.num_train_epochs):
            for step, batch in enumerate(train_data_loader):
                input_ids, segment_ids, labels, example_ids = batch
                logits = model(input_ids=input_ids, token_type_ids=segment_ids)
                loss = loss_fct(logits, labels)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    global_step += 1
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.clear_grad()
                    if global_step % args.logging_steps == 0:
                        logger.info(
                            "global step %d/%d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                            % (global_step, num_training_steps, epoch, step + 1,
                               loss, args.logging_steps /
                               (time.time() - tic_train)))
                        tic_train = time.time()
                if global_step >= num_training_steps:
                    break
            if global_step > num_training_steps:
                break
            tic_eval = time.time()
            acc = evaluate(model, dev_data_loader)
            logger.info("eval acc: %.5f, eval done total : %s s" %
                        (acc, time.time() - tic_eval))
            if paddle.distributed.get_rank() == 0 and acc > best_acc:
                best_acc = acc
                if args.save_best_model:
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    if not os.path.exists(args.output_dir):
                        os.makedirs(args.output_dir)
                    model_to_save.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
            if global_step >= num_training_steps:
                break
        logger.info("best_result: %.2f" % (best_acc * 100))

    if args.do_predict:
        column_names = test_ds.column_names
        test_ds = test_ds.map(partial(preprocess_function, do_predict=True),
                              batched=True,
                              batch_size=len(test_ds),
                              remove_columns=column_names,
                              num_proc=args.num_proc)
        test_batch_sampler = paddle.io.BatchSampler(
            test_ds, batch_size=args.eval_batch_size, shuffle=False)

        batchify_fn = lambda samples, fn=Dict({
            'input_ids':
            Pad(axis=1, pad_val=tokenizer.pad_token_id),  # input
            'token_type_ids':
            Pad(axis=1, pad_val=tokenizer.pad_token_type_id),  # segment
            'example_ids':
            Stack(dtype="int64"),  # example id
        }): fn(samples)

        test_data_loader = paddle.io.DataLoader(
            dataset=test_ds,
            batch_sampler=test_batch_sampler,
            collate_fn=batchify_fn,
            return_list=True)

        result = {}
        idx = 623377
        preds = evaluate(model, test_data_loader, do_predict=True)
        for pred in preds:
            result["#idiom" + str(idx) + "#"] = pred
            idx += 1
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, 'chid11_predict.json'),
                  "w") as writer:
            json.dump(result, writer, indent=2)


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    run(args)
