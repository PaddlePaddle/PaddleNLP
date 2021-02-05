# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import os
import random
import time
import math
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader

import paddlenlp as ppnlp
from paddlenlp.datasets import MSRA_NER
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import BertForTokenClassification, BertTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import ChunkEvaluator

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
    help="Path to pre-trained model or shortcut name selected in the list: " +
    ", ".join(list(BertTokenizer.pretrained_init_configuration.keys())))
parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    required=True,
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument(
    "--max_seq_length",
    default=128,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument(
    "--batch_size",
    default=8,
    type=int,
    help="Batch size per GPU/CPU for training.", )
parser.add_argument(
    "--learning_rate",
    default=5e-5,
    type=float,
    help="The initial learning rate for Adam.")
parser.add_argument(
    "--weight_decay",
    default=0.0,
    type=float,
    help="Weight decay if we apply some.")
parser.add_argument(
    "--adam_epsilon",
    default=1e-8,
    type=float,
    help="Epsilon for Adam optimizer.")
parser.add_argument(
    "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument(
    "--num_train_epochs",
    default=3,
    type=int,
    help="Total number of training epochs to perform.", )
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
)
parser.add_argument(
    "--warmup_steps",
    default=0,
    type=int,
    help="Linear warmup over warmup_steps.")

parser.add_argument(
    "--logging_steps", type=int, default=1, help="Log every X updates steps.")
parser.add_argument(
    "--save_steps",
    type=int,
    default=100,
    help="Save checkpoint every X updates steps.")
parser.add_argument(
    "--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument(
    "--n_gpu", type=int, default=1, help="number of gpus to use, 0 for cpu.")


def evaluate(model, loss_fct, metric, data_loader, label_num):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, segment_ids, length, labels = batch
        logits = model(input_ids, segment_ids)
        loss = loss_fct(logits.reshape([-1, label_num]), labels.reshape([-1]))
        avg_loss = paddle.mean(loss)
        preds = logits.argmax(axis=2)
        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(
            None, length, preds, labels)
        metric.update(num_infer_chunks.numpy(),
                      num_label_chunks.numpy(), num_correct_chunks.numpy())
        precision, recall, f1_score = metric.accumulate()
    print("eval loss: %f, precision: %f, recall: %f, f1: %f" %
          (avg_loss, precision, recall, f1_score))
    model.train()


def convert_example(example,
                    tokenizer,
                    label_list,
                    no_entity_id,
                    max_seq_length=512,
                    is_test=False):
    """convert a glue example into necessary features"""

    def _truncate_seqs(seqs, max_seq_length):
        if len(seqs) == 1:  # single sentence
            # Account for [CLS] and [SEP] with "- 2"
            seqs[0] = seqs[0][0:(max_seq_length - 2)]
        else:  # sentence pair
            # Account for [CLS], [SEP], [SEP] with "- 3"
            tokens_a, tokens_b = seqs
            max_seq_length -= 3
            while True:  # truncate with longest_first strategy
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_seq_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()
        return seqs

    def _concat_seqs(seqs, separators, seq_mask=0, separator_mask=1):
        concat = sum((seq + sep for sep, seq in zip(separators, seqs)), [])
        segment_ids = sum(
            ([i] * (len(seq) + len(sep))
             for i, (sep, seq) in enumerate(zip(separators, seqs))), [])
        if isinstance(seq_mask, int):
            seq_mask = [[seq_mask] * len(seq) for seq in seqs]
        if isinstance(separator_mask, int):
            separator_mask = [[separator_mask] * len(sep) for sep in separators]
        p_mask = sum((s_mask + mask
                      for sep, seq, s_mask, mask in zip(
                          separators, seqs, seq_mask, separator_mask)), [])
        return concat, segment_ids, p_mask

    def _reseg_token_label(tokens, tokenizer, labels=None):
        if labels:
            if len(tokens) != len(labels):
                raise ValueError(
                    "The length of tokens must be same with labels")
            ret_tokens = []
            ret_labels = []
            for token, label in zip(tokens, labels):
                sub_token = tokenizer(token)
                if len(sub_token) == 0:
                    continue
                ret_tokens.extend(sub_token)
                ret_labels.append(label)
                if len(sub_token) < 2:
                    continue
                sub_label = label
                if label.startswith("B-"):
                    sub_label = "I-" + label[2:]
                ret_labels.extend([sub_label] * (len(sub_token) - 1))

            if len(ret_tokens) != len(ret_labels):
                raise ValueError(
                    "The length of ret_tokens can't match with labels")
            return ret_tokens, ret_labels
        else:
            ret_tokens = []
            for token in tokens:
                sub_token = tokenizer(token)
                if len(sub_token) == 0:
                    continue
                ret_tokens.extend(sub_token)
                if len(sub_token) < 2:
                    continue

            return ret_tokens, None

    if not is_test:
        # get the label
        label = example[-1].split("\002")
        example = example[0].split("\002")
        #create label maps if classification task
        label_map = {}
        for (i, l) in enumerate(label_list):
            label_map[l] = i
    else:
        label = None

    tokens_raw, labels_raw = _reseg_token_label(
        tokens=example, labels=label, tokenizer=tokenizer)
    # truncate to the truncate_length,
    tokens_trun = _truncate_seqs([tokens_raw], max_seq_length)
    # concate the sequences with special tokens
    tokens_trun[0] = [tokenizer.cls_token] + tokens_trun[0]
    tokens, segment_ids, _ = _concat_seqs(tokens_trun, [[tokenizer.sep_token]] *
                                          len(tokens_trun))
    # convert the token to ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    valid_length = len(input_ids)
    if labels_raw:
        labels_trun = _truncate_seqs([labels_raw], max_seq_length)[0]
        labels_id = [no_entity_id] + [label_map[lbl]
                                      for lbl in labels_trun] + [no_entity_id]
    if not is_test:
        return input_ids, segment_ids, valid_length, labels_id
    else:
        return input_ids, segment_ids, valid_length


def do_train(args):
    paddle.set_device("gpu" if args.n_gpu else "cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    train_dataset, test_dataset = ppnlp.datasets.MSRA_NER.get_datasets(
        ["train", "test"])
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    label_list = train_dataset.get_labels()
    label_num = len(label_list)
    no_entity_id = label_num - 1
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=label_list,
        no_entity_id=label_num - 1,
        max_seq_length=args.max_seq_length)
    train_dataset = train_dataset.apply(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    ignore_label = -100
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # input
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # segment
        Stack(),  # length
        Pad(axis=0, pad_val=ignore_label)  # label
    ): fn(samples)
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    test_dataset = test_dataset.apply(trans_func, lazy=True)
    dev_batch_sampler = paddle.io.BatchSampler(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    model = BertForTokenClassification.from_pretrained(
        args.model_name_or_path, num_classes=label_num)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = args.max_steps if args.max_steps > 0 else len(
        train_data_loader) * args.num_train_epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_steps)

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ])

    loss_fct = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)

    metric = ChunkEvaluator(label_list=train_dataset.get_labels())

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, segment_ids, length, labels = batch
            logits = model(input_ids, segment_ids)
            loss = loss_fct(
                logits.reshape([-1, label_num]), labels.reshape([-1]))
            avg_loss = paddle.mean(loss)
            if global_step % args.logging_steps == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_step, epoch, step, avg_loss,
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            avg_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_gradients()
            if global_step % args.save_steps == 0:
                evaluate(model, loss_fct, metric, test_data_loader, label_num)
                if (not args.n_gpu > 1) or paddle.distributed.get_rank() == 0:
                    paddle.save(model.state_dict(),
                                os.path.join(args.output_dir,
                                             "model_%d.pdparams" % global_step))
    # Save final model 
    if (global_step) % args.save_steps != 0:
        evaluate(model, loss_fct, metric, test_data_loader, label_num)
        if (not args.n_gpu > 1) or paddle.distributed.get_rank() == 0:
            paddle.save(model.state_dict(),
                        os.path.join(args.output_dir,
                                     "model_%d.pdparams" % global_step))


if __name__ == "__main__":
    args = parser.parse_args()
    if args.n_gpu > 1:
        paddle.distributed.spawn(do_train, args=(args, ), nprocs=args.n_gpu)
    else:
        do_train(args)
