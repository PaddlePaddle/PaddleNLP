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

import argparse
import collections
from collections import namedtuple, defaultdict

import os
import random
from functools import partial
import time

import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from paddle.optimizer import AdamW

from paddlenlp.transformers import ErnieDocForTokenClassification
from paddlenlp.transformers import ErnieDocTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.log import logger
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.ops.optimizer import layerwise_lr_decay

from data import SequenceLabelingIterator

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--model_name_or_path", type=str, default="ernie-doc-base-zh", help="Pretraining model name or path")
parser.add_argument("--max_seq_length", type=int, default=512, help="The maximum total input sequence length after SentencePiece tokenization.")
parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate used to train.")
parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
parser.add_argument("--logging_steps", type=int, default=1, help="Log every X updates steps.")
parser.add_argument("--output_dir", type=str, default='checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--epochs", type=int, default=3, help="Number of epoches for training.")
parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Select cpu, gpu devices to train model.")
parser.add_argument("--seed", type=int, default=1, help="Random seed for initialization.")
parser.add_argument("--memory_length", type=int, default=128, help="Length of the retained previous heads.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Linear warmup proption over the training process.")
parser.add_argument("--dataset", default="msra_ner", choices=["msra_ner"], type=str, help="The training dataset")
parser.add_argument("--layerwise_decay", default=1.0, type=float, help="Layerwise decay ratio")
parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)

# yapf: enable
args = parser.parse_args()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def init_memory(batch_size, memory_length, d_model, n_layers):
    return [
        paddle.zeros([batch_size, memory_length, d_model], dtype="float32")
        for _ in range(n_layers)
    ]


@paddle.no_grad()
def evaluate(model, metric, data_loader, memories0):
    model.eval()
    metric.reset()
    avg_loss, precision, recall, f1_score = 0, 0, 0, 0
    loss_fct = paddle.nn.loss.CrossEntropyLoss()
    losses = []
    # Copy the memory
    memories = list(memories0)
    tic_train = time.time()
    eval_logging_step = 500
    labels_dict = defaultdict(list)
    preds_dict = defaultdict(list)
    length_dict = defaultdict(list)

    for step, batch in enumerate(data_loader, start=1):
        input_ids, position_ids, token_type_ids, attn_mask, labels, lengths, qids, \
            gather_idxs, need_cal_loss = batch
        logits, memories = model(input_ids, memories, token_type_ids,
                                 position_ids, attn_mask)
        logits, labels, qids, lengths = list(
            map(lambda x: paddle.gather(x, gather_idxs),
                [logits, labels, qids, lengths]))
        loss = loss_fct(logits, labels)
        avg_loss = loss.mean()
        losses.append(avg_loss)
        preds = logits.argmax(axis=2)

        np_qids = qids.numpy().flatten()
        for i, qid in enumerate(np_qids):
            preds_dict[qid].append(preds[i])
            labels_dict[qid].append(labels[i])
            length_dict[qid].append(lengths[i])

        if step % eval_logging_step == 0:
            logger.info("Step %d: loss:  %.5f, speed: %.5f steps/s" %
                        (step, np.mean(losses), eval_logging_step /
                         (time.time() - tic_train)))
            tic_train = time.time()

    qids = preds_dict.keys()
    for qid in qids:
        preds = paddle.concat(preds_dict[qid], axis=0).unsqueeze(0)
        labels = paddle.concat(labels_dict[qid],
                               axis=0).unsqueeze(0).squeeze(-1)
        length = paddle.concat(length_dict[qid], axis=0)
        length = length.sum(axis=0)
        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(
            length, preds, labels)
        metric.update(num_infer_chunks.numpy(), num_label_chunks.numpy(),
                      num_correct_chunks.numpy())
    precision, recall, f1_score = metric.accumulate()
    metric.reset()
    logger.info("Total {} samples.".format(len(qids)))
    logger.info("eval loss: %f, precision: %f, recall: %f, f1: %f" %
                (avg_loss, precision, recall, f1_score))
    model.train()
    return precision, recall, f1_score


def do_train(args):
    set_seed(args)
    tokenizer = ErnieDocTokenizer.from_pretrained(args.model_name_or_path)
    train_ds, eval_ds = load_dataset(args.dataset, splits=["train", "test"])
    test_ds = eval_ds

    num_classes = len(train_ds.label_list)
    no_entity_id = num_classes - 1

    paddle.set_device(args.device)
    trainer_num = paddle.distributed.get_world_size()
    if trainer_num > 1:
        paddle.distributed.init_parallel_env()
    rank = paddle.distributed.get_rank()
    if rank == 0:
        if os.path.exists(args.model_name_or_path):
            logger.info("init checkpoint from %s" % args.model_name_or_path)
    model = ErnieDocForTokenClassification.from_pretrained(
        args.model_name_or_path, num_classes=num_classes)
    model_config = model.ernie_doc.config
    if trainer_num > 1:
        model = paddle.DataParallel(model)

    train_ds_iter = SequenceLabelingIterator(
        train_ds,
        args.batch_size,
        tokenizer,
        trainer_num,
        trainer_id=rank,
        memory_len=model_config["memory_len"],
        max_seq_length=args.max_seq_length,
        random_seed=args.seed,
        no_entity_id=no_entity_id)
    eval_ds_iter = SequenceLabelingIterator(
        eval_ds,
        args.batch_size,
        tokenizer,
        trainer_num,
        trainer_id=rank,
        memory_len=model_config["memory_len"],
        max_seq_length=args.max_seq_length,
        mode="eval",
        no_entity_id=no_entity_id)
    test_ds_iter = SequenceLabelingIterator(
        test_ds,
        args.batch_size,
        tokenizer,
        trainer_num,
        trainer_id=rank,
        memory_len=model_config["memory_len"],
        max_seq_length=args.max_seq_length,
        mode="test",
        no_entity_id=no_entity_id)

    train_dataloader = paddle.io.DataLoader.from_generator(capacity=70,
                                                           return_list=True)
    train_dataloader.set_batch_generator(train_ds_iter, paddle.get_device())
    eval_dataloader = paddle.io.DataLoader.from_generator(capacity=70,
                                                          return_list=True)
    eval_dataloader.set_batch_generator(eval_ds_iter, paddle.get_device())
    test_dataloader = paddle.io.DataLoader.from_generator(capacity=70,
                                                          return_list=True)
    test_dataloader.set_batch_generator(test_ds_iter, paddle.get_device())

    num_training_examples = train_ds_iter.get_num_examples()
    num_training_steps = args.epochs * num_training_examples // args.batch_size // trainer_num
    logger.info("Device count: %d, trainer_id: %d" % (trainer_num, rank))
    logger.info("Num train examples: %d" % num_training_examples)
    logger.info("Max train steps: %d" % num_training_steps)
    logger.info("Num warmup steps: %d" %
                int(num_training_steps * args.warmup_proportion))

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    # Construct dict
    name_dict = dict()
    for n, p in model.named_parameters():
        name_dict[p.name] = n

    simple_lr_setting = partial(layerwise_lr_decay, args.layerwise_decay,
                                name_dict, model_config["num_hidden_layers"])

    optimizer = AdamW(learning_rate=lr_scheduler,
                      parameters=model.parameters(),
                      weight_decay=args.weight_decay,
                      apply_decay_param_fun=lambda x: x in decay_params,
                      lr_ratio=simple_lr_setting)

    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = ChunkEvaluator(label_list=train_ds.label_list)

    global_steps = 0

    create_memory = partial(init_memory, args.batch_size, args.memory_length,
                            model_config["hidden_size"],
                            model_config["num_hidden_layers"])
    # Copy the memory
    memories = create_memory()
    tic_train = time.time()
    best_f1 = 0
    stop_training = False
    for epoch in range(args.epochs):
        train_ds_iter.shuffle_sample()
        train_dataloader.set_batch_generator(train_ds_iter, paddle.get_device())
        for step, batch in enumerate(train_dataloader, start=1):
            global_steps += 1
            input_ids, position_ids, token_type_ids, attn_mask, labels, lengths, qids, \
                gather_idx, need_cal_loss = batch
            logits, memories = model(input_ids, memories, token_type_ids,
                                     position_ids, attn_mask)
            logits, labels = list(
                map(lambda x: paddle.gather(x, gather_idx), [logits, labels]))

            loss = criterion(logits, labels) * need_cal_loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_steps % args.logging_steps == 0:
                logger.info(
                    "train: global step %d, epoch: %d, loss: %f, lr: %f, speed: %.2f step/s"
                    % (global_steps, epoch, loss, lr_scheduler.get_lr(),
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            if global_steps % args.save_steps == 0:
                # Evaluate
                logger.info("Eval:")
                precision, recall, f1_score = evaluate(model, metric,
                                                       eval_dataloader,
                                                       create_memory())
                # Save
                if rank == 0:
                    output_dir = os.path.join(args.output_dir,
                                              "model_%d" % (global_steps))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    if f1_score > best_f1:
                        logger.info("Save best model......")
                        best_f1 = f1_score
                        best_model_dir = os.path.join(args.output_dir,
                                                      "best_model")
                        if not os.path.exists(best_model_dir):
                            os.makedirs(best_model_dir)
                        model_to_save.save_pretrained(best_model_dir)
                        tokenizer.save_pretrained(best_model_dir)

            if args.max_steps > 0 and global_steps >= args.max_steps:
                stop_training = True
                break
        if stop_training:
            break

    logger.info("Final test result:")
    eval_acc = evaluate(model, metric, test_dataloader, create_memory())


if __name__ == "__main__":
    do_train(args)
