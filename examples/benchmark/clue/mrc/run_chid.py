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
import random
from functools import partial

import numpy as np

import paddle
from paddle.metric import Accuracy
import paddle.nn as nn

from datasets import load_dataset

from paddlenlp.data import Pad, Stack, Tuple, Dict
from paddlenlp.transformers import AutoModelForMultipleChoice, AutoTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name.")
    parser.add_argument(
        "--output_dir",
        default="best_clue_model",
        type=str,
        help="The  path of the checkpoints .", )
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps. If > 0: Override warmup_proportion"
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Linear warmup proportion over total steps.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-6,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="The max value of grad norm.")
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--eval_batch_size",
        default=24,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumualte before performing a backward/update pass."
    )
    parser.add_argument(
        "--do_train", action='store_true', help="Whether to train.")
    parser.add_argument(
        "--do_predict", action='store_true', help="Whether to predict.")
    parser.add_argument(
        "--max_seq_length",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--logging_steps",
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
def evaluate(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    for step, batch in enumerate(data_loader):
        input_ids, segment_ids, labels = batch
        logits = model(input_ids=input_ids, token_type_ids=segment_ids)
        loss = loss_fct(logits, labels)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    model.train()
    return res


def run(args):
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
            result = {"input_ids": [], "token_type_ids": []}
        else:
            result = {"input_ids": [], "token_type_ids": [], "labels": []}
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

                    tmp_l, tmp_r = add_tokens_for_around(all_doc_tokens, pos,
                                                         num_tokens)
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
                    if not do_predict:
                        label = examples.data["answers"][idx]["candidate_id"][
                            candidate]
                        result["labels"].append(label)
                    candidate += 1
            if (idx + 1) % 10000 == 0:
                print(idx + 1, "samples have been processed.")
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
        train_ds = train_ds.map(partial(preprocess_function),
                                batched=True,
                                batch_size=len(train_ds),
                                num_proc=1,
                                remove_columns=column_names)
        batchify_fn = lambda samples, fn=Dict({
            'input_ids': Pad(axis=1, pad_val=tokenizer.pad_token_id),  # input
            'token_type_ids': Pad(axis=1, pad_val=tokenizer.pad_token_type_id),  # segment
            'labels': Stack(dtype="int64")  # label
        }): fn(samples)

        train_batch_sampler = paddle.io.DistributedBatchSampler(
            train_ds, batch_size=args.batch_size, shuffle=True)
        train_data_loader = paddle.io.DataLoader(
            dataset=train_ds,
            batch_sampler=train_batch_sampler,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)

        dev_ds = dev_ds.map(partial(preprocess_function),
                            batched=True,
                            batch_size=len(dev_ds),
                            remove_columns=column_names,
                            num_proc=1)

        dev_batch_sampler = paddle.io.BatchSampler(
            dev_ds, batch_size=args.eval_batch_size, shuffle=False)

        dev_data_loader = paddle.io.DataLoader(
            dataset=dev_ds,
            batch_sampler=dev_batch_sampler,
            collate_fn=batchify_fn,
            return_list=True)

        num_training_steps = int(
            len(train_data_loader) * args.num_train_epochs /
            args.gradient_accumulation_steps)
        lr_scheduler = LinearDecayWithWarmup(args.learning_rate,
                                             num_training_steps, 0)
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
        metric = Accuracy()

        model.train()
        global_step = 0
        best_acc = 0.0
        tic_train = time.time()
        for epoch in range(args.num_train_epochs):
            for step, batch in enumerate(train_data_loader):
                input_ids, segment_ids, labels = batch
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
                        print(
                            "global step %d/%d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                            % (global_step, num_training_steps, epoch, step + 1,
                               loss,
                               args.logging_steps / (time.time() - tic_train)))
                        tic_train = time.time()
            tic_eval = time.time()
            acc = evaluate(model, loss_fct, metric, dev_data_loader)
            print("eval acc: %.5f, eval done total : %s s" %
                  (acc, time.time() - tic_eval))
            if paddle.distributed.get_rank() == 0 and acc > best_acc:
                best_acc = acc
                model_to_save = model._layers if isinstance(
                    model, paddle.DataParallel) else model
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                model_to_save.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
        print("best_acc: ", best_acc)

    if args.do_predict:
        column_names = test_ds.column_names
        test_ds = test_ds.map(partial(
            preprocess_function, do_predict=True),
                              batched=True,
                              batch_size=len(test_ds),
                              remove_columns=column_names,
                              num_proc=1)

        test_batch_sampler = paddle.io.BatchSampler(
            test_ds, batch_size=args.eval_batch_size, shuffle=False)

        batchify_fn = lambda samples, fn=Dict({
            'input_ids': Pad(axis=1, pad_val=tokenizer.pad_token_id),  # input
            'token_type_ids': Pad(axis=1, pad_val=tokenizer.pad_token_type_id),  # segment
        }): fn(samples)

        test_data_loader = paddle.io.DataLoader(
            dataset=test_ds,
            batch_sampler=test_batch_sampler,
            collate_fn=batchify_fn,
            return_list=True)

        result = {}
        idx = 623377
        for step, batch in enumerate(test_data_loader):
            input_ids, segment_ids = batch
            with paddle.no_grad():
                logits = model(input_ids, segment_ids)
            preds = paddle.argmax(logits, axis=1).numpy().tolist()
            for pred in preds:
                result["#idiom" + str(idx)] = pred
                idx += 1

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(
                os.path.join(args.output_dir, 'chid11_predict.json'),
                "w",
                encoding='utf-8') as writer:
            writer.write(
                json.dumps(
                    result, ensure_ascii=False, indent=4) + "\n")


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
