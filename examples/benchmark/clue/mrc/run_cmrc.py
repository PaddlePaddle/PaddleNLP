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
import random
import time
import json
import math
import distutils.util
import argparse
import contextlib

from functools import partial
import numpy as np
import paddle

from paddle.io import DataLoader

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.transformers import AutoModelForQuestionAnswering, AutoTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction
from paddlenlp.utils.log import logger

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name of model.")
    parser.add_argument(
        "--output_dir",
        default="best_cmrc_model",
        type=str,
        help=
        "The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument("--save_best_model",
                        default=True,
                        type=distutils.util.strtobool,
                        help="Whether to save best model.")
    parser.add_argument("--overwrite_cache",
                        default=False,
                        type=distutils.util.strtobool,
                        help="Whether to overwrite cache for dataset.")
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--batch_size",
                        default=8,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--num_proc",
        default=None,
        type=int,
        help=
        "Max number of processes when generating cache. Already cached shards are loaded sequentially."
    )
    parser.add_argument("--eval_batch_size",
                        default=12,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs."
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help=
        "Linear warmup over warmup_steps. If > 0: Override warmup_proportion")
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help=
        "Proportion of training steps to perform linear learning rate warmup for."
    )
    parser.add_argument("--logging_steps",
                        type=int,
                        default=100,
                        help="Log every X updates steps.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default="gpu",
        help="Select which device to train model, defaults to gpu.")
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help=
        "When splitting up a long document into chunks, how much stride to take between chunks."
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help=
        "The total number of n-best predictions to generate in the nbest_predictions.json output file."
    )
    parser.add_argument("--max_query_length",
                        type=int,
                        default=64,
                        help="Max query length.")
    parser.add_argument("--max_answer_length",
                        type=int,
                        default=50,
                        help="Max answer length.")
    parser.add_argument(
        "--do_lower_case",
        action='store_false',
        help=
        "Whether to lower case the input text. Should be True for uncased models and False for cased models."
    )
    parser.add_argument("--verbose",
                        action='store_true',
                        help="Whether to output verbose log.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to train.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to predict.")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=2,
        help=
        "Number of updates steps to accumualte before performing a backward/update pass."
    )
    args = parser.parse_args()
    return args


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


@paddle.no_grad()
def evaluate(model, raw_dataset, dataset, data_loader, args, do_eval=True):
    model.eval()

    all_start_logits = []
    all_end_logits = []
    tic_eval = time.time()
    for batch in data_loader:
        start_logits, end_logits = model(**batch)
        for idx in range(start_logits.shape[0]):
            if len(all_start_logits) % 1000 == 0 and len(all_start_logits):
                logger.info("Processing example: %d" % len(all_start_logits))
                logger.info('time per 1000: %s' % (time.time() - tic_eval))
                tic_eval = time.time()

            all_start_logits.append(start_logits.numpy()[idx])
            all_end_logits.append(end_logits.numpy()[idx])

    all_predictions, _, _ = compute_prediction(
        raw_dataset, dataset, (all_start_logits, all_end_logits), False,
        args.n_best_size, args.max_answer_length)

    mode = 'validation' if do_eval else 'test'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if do_eval:
        filename = os.path.join(args.output_dir, 'prediction_validation.json')
    else:
        filename = os.path.join(args.output_dir, 'cmrc2018_predict.json')
    with open(filename, "w", encoding='utf-8') as writer:
        writer.write(
            json.dumps(all_predictions, ensure_ascii=False, indent=4) + "\n")
    if do_eval:
        res = squad_evaluate(examples=[raw_data for raw_data in raw_dataset],
                             preds=all_predictions,
                             is_whitespace_splited=False)
        model.train()
        return res['exact'], res['f1']

    model.train()


class CrossEntropyLossForSQuAD(paddle.nn.Layer):

    def __init__(self):
        super(CrossEntropyLossForSQuAD, self).__init__()

    def forward(self, y, label):
        start_logits, end_logits = y
        start_position, end_position = label
        start_position = paddle.unsqueeze(start_position, axis=-1)
        end_position = paddle.unsqueeze(end_position, axis=-1)
        start_loss = paddle.nn.functional.cross_entropy(input=start_logits,
                                                        label=start_position)
        end_loss = paddle.nn.functional.cross_entropy(input=end_logits,
                                                      label=end_position)
        loss = (start_loss + end_loss) / 2
        return loss


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
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    rank = paddle.distributed.get_rank()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    set_seed(args)

    train_examples, dev_examples, test_examples = load_dataset(
        'clue', 'cmrc2018', split=["train", "validation", "test"])

    column_names = train_examples.column_names
    if rank == 0:
        if os.path.exists(args.model_name_or_path):
            logger.info("init checkpoint from %s" % args.model_name_or_path)

    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        # NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
        # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
        contexts = examples['context']
        questions = examples['question']

        tokenized_examples = tokenizer(questions,
                                       contexts,
                                       stride=args.doc_stride,
                                       max_seq_len=args.max_seq_length)

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples['token_type_ids'][i]

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples['answers'][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1
                token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char
                        and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[
                            token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(
                        token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index +
                                                               1)

        return tokenized_examples

    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        #NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
        # that HuggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
        contexts = examples['context']
        questions = examples['question']

        tokenized_examples = tokenizer(questions,
                                       contexts,
                                       stride=args.doc_stride,
                                       max_seq_len=args.max_seq_length,
                                       return_attention_mask=True)

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples['token_type_ids'][i]
            context_index = 1

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(
                examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index
                 and k != len(sequence_ids) - 1 else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    if args.do_train:
        args.batch_size = int(args.batch_size /
                              args.gradient_accumulation_steps)

        with main_process_first(desc="train dataset map pre-processing"):
            train_ds = train_examples.map(
                prepare_train_features,
                batched=True,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                num_proc=args.num_proc,
                desc="Running tokenizer on train dataset")
        train_batch_sampler = paddle.io.DistributedBatchSampler(
            train_ds, batch_size=args.batch_size, shuffle=True)

        batchify_fn = DataCollatorWithPadding(tokenizer)
        train_data_loader = DataLoader(dataset=train_ds,
                                       batch_sampler=train_batch_sampler,
                                       collate_fn=batchify_fn,
                                       return_list=True)

        with main_process_first(desc="evaluate dataset map pre-processing"):
            dev_ds = dev_examples.map(
                prepare_validation_features,
                batched=True,
                remove_columns=column_names,
                num_proc=args.num_proc,
                load_from_cache_file=args.overwrite_cache,
                desc="Running tokenizer on validation dataset")
        dev_ds_for_model = dev_ds.remove_columns(
            ["example_id", "offset_mapping", "attention_mask"])
        dev_batch_sampler = paddle.io.BatchSampler(
            dev_ds, batch_size=args.eval_batch_size, shuffle=False)

        dev_data_loader = DataLoader(dataset=dev_ds_for_model,
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
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            epsilon=args.adam_epsilon,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params)
        criterion = CrossEntropyLossForSQuAD()
        best_res = (0.0, 0.0)
        global_step = 0
        tic_train = time.time()
        for epoch in range(args.num_train_epochs):
            for step, batch in enumerate(train_data_loader):
                start_positions = batch.pop("start_positions")
                end_positions = batch.pop("end_positions")
                logits = model(**batch)
                loss = criterion(logits, (start_positions, end_positions))
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
                            "global step %d/%d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                            % (global_step, num_training_steps, epoch, step + 1,
                               loss, args.logging_steps /
                               (time.time() - tic_train)))
                        tic_train = time.time()
                    if global_step >= num_training_steps:
                        logger.info("best_result: %.2f/%.2f" %
                                    (best_res[0], best_res[1]))
                        return
            em, f1 = evaluate(model, dev_examples, dev_ds, dev_data_loader,
                              args)
            if paddle.distributed.get_rank() == 0 and em > best_res[0]:
                best_res = (em, f1)
                if args.save_best_model:
                    output_dir = args.output_dir
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # need better way to get inner model of DataParallel
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
        logger.info("best_result: %.2f/%.2f" % (best_res[0], best_res[1]))

    if args.do_predict and rank == 0:
        test_ds = test_examples.map(prepare_validation_features,
                                    batched=True,
                                    remove_columns=column_names,
                                    num_proc=args.num_proc)
        test_ds_for_model = test_ds.remove_columns(
            ["example_id", "offset_mapping", "attention_mask"])
        dev_batchify_fn = DataCollatorWithPadding(tokenizer)

        test_batch_sampler = paddle.io.BatchSampler(
            test_ds_for_model, batch_size=args.eval_batch_size, shuffle=False)

        batchify_fn = DataCollatorWithPadding(tokenizer)
        test_data_loader = DataLoader(dataset=test_ds_for_model,
                                      batch_sampler=test_batch_sampler,
                                      collate_fn=batchify_fn,
                                      return_list=True)

        evaluate(model,
                 test_examples,
                 test_ds,
                 test_data_loader,
                 args,
                 do_eval=False)


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
