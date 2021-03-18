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

import os
import argparse
import pathlib
import time
import random
from functools import partial

import paddle
from paddle.io import DataLoader
import json
import numpy as np
from paddlenlp.utils.log import logger
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import ErnieCtmWordtagModel, ErnieCtmTokenizer, LinearDecayWithWarmup
from paddlenlp.metrics import ChunkEvaluator

# from .data import load_labels, WordTagDataset
from metrics import Accuracy, AverageMeter, SequenceAccuracy


def load_dataset(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                example = json.loads(line)
                words = example["tokens"]
                tags = example["tags"]
                cls_label = example["cls_label"]
                yield words, tags, cls_label

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]


def load_dict(dict_path):
    vocab = {}
    i = 0
    for line in open(dict_path, 'r', encoding='utf-8'):
        vocab[line.strip()] = i
        i += 1
    return vocab


def convert_example(example,
                    tokenizer,
                    max_seq_len,
                    tags_to_idx,
                    labels_to_idx,
                    summary_num=2):
    words, tags, cls_label = example
    tokens = [f"[CLS{i}]" for i in range(1, summary_num)] + words
    tokenized_input = tokenizer(
        tokens,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)
    # '[CLS]' and '[SEP]' will get label 'O'
    tags = ['O'] * (summary_num) + tags + ['O']
    tokenized_input['tags'] = [tags_to_idx[x] for x in tags]
    tokenized_input['cls_label'] = labels_to_idx['cls_label']
    return tokenized_input['input_ids'], tokenized_input[
        'token_type_ids'], tokenized_input['seq_len'], tokenized_input[
            'tags'], tokenized_input['cls_label']


def parse_args():
    parser = argparse.ArgumentParser()

    # yapf: disable
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir, should contain [train/test].json and [train/test]_metrics.json")
    parser.add_argument("--model_dir", default="ernie-ctm", type=str, help="The pre-trained model checkpoint dir.")
    parser.add_argument("--output_dir",default="./outpout_dir",type=str, help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--max_seq_len",default=128,type=int,help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--learning_rate",default=1e-4,type=float,help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",default=3,type=int,help="Total number of training epochs to perform.", )
    parser.add_argument("--logging_steps",type=int,default=100,help="Log every X updates steps.")
    parser.add_argument("--save_steps",type=int,default=100,help="Save checkpoint every X updates steps.")
    parser.add_argument("--batch_size",default=32,type=int,help="Batch size per GPU/CPU for training.", )
    parser.add_argument("--weight_decay",default=0.01,type=float,help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps",default=0,type=int,help="Linear warmup over warmup_steps. If > 0: Override warmup_proportion")
    parser.add_argument("--warmup_proportion",default=0.1,type=float,help="Linear warmup proportion over total steps.")
    parser.add_argument("--adam_epsilon",default=1e-6,type=float,help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_steps",default=-1,type=int,help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument("--n_gpu",default=1,type=int,help="number of gpus to use, 0 for cpu.")
    # yapf: enable

    # parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    # parser.add_argument("--local_rank", default=-1, type=int)
    # parser.add_argument("--fp16", action="store_true")

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


# @paddle.no_grad()
# def evaluate(model, cls_acc, tr_seq_acc, test_data_loader, labels_to_idx, vit):
#     model.eval()
#     cls_acc.reset()
#     seq_acc, total_len = 0.0, 0.0
#     for batch in test_data_loader:
#         input_ids, token_type_ids, attention_mask, tags, length, labels = batch
#         inputs = {
#             "input_ids": input_ids,
#             "token_type_ids": token_type_ids,
#             "attention_mask": attention_mask,
#             "length": length,
#         }
#         outputs = model(**inputs)
#         cls_acc(logits=outputs[0].view(-1, len(labels_to_idx)), target=labels.view(-1))
#         pred_tags = vit(outputs[1], masks)
#         pred_cls_label = paddle.argmax(outputs[0], dim=-1, keepdim=False)
#         for i, pred_tag in enumerate(pred_tags):
#             if pred_cls_label[i] != labels_to_idx["其他文本"]:
#                 continue
#             total_len += float(len(pred_tag[0]))
#             for j, tag in enumerate(pred_tag[0]):
#                 if tag == tags[i, j]:
#                     seq_acc += 1.0
#         tr_seq_acc.update(seq_acc, n=1)
#         tr_seq_acc.count += total_len - 1
#     logger.info("Classification Accuracy: %s,  Sequence Labeling Accuracy: %s"%(cls_acc.value(), tr_seq_acc.avg))
#     tr_seq_acc.reset()
#     cls_acc.reset()
#     model.train()


def do_train(args):
    paddle.set_device("gpu" if args.n_gpu else "cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    set_seed(args)

    # output_dir = pathlib.Path(args.output_dir)

    # Loading data.

    # train_ds = WordTagDataset.from_file(pathlib.Path(args.data_dir), "train", tokenizer,
    #                                          labels_to_idx=labels_to_idx, tags_to_idx=tags_to_idx)
    # test_ds = WordTagDataset.from_file(pathlib.Path(args.data_dir), "test", tokenizer,
    #                                         labels_to_idx=labels_to_idx, tags_to_idx=tags_to_idx)
    train_ds, test_ds = load_dataset(datafiles=('./data/train.json',
                                                './data/test.json'))
    tags_to_idx = load_dict("./data/tags.txt")
    labels_to_idx = load_dict("./data/classifier_labels.txt")
    tokenizer = ErnieCtmTokenizer.from_pretrained(args.model_dir)
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        tags_to_idx=tags_to_idx,
        labels_to_idx=labels_to_idx)
    train_ds.map(trans_func)
    test_ds.map(trans_func)

    ignore_label = tags_to_idx["O"]
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
        Stack(),  # seq_len
        Pad(axis=0, pad_val=ignore_label),  # tags
        Stack(),  # cls_label
    ): fn(samples)

    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_data_loader = DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        num_workers=0,
        collate_fn=batchify_fn,
        return_list=True)
    test_data_loader = DataLoader(
        test_ds,
        collate_fn=batchify_fn,
        num_workers=0,
        batch_size=args.batch_size,
        shuffle=False,
        return_list=True)

    model = ErnieCtmWordtagModel.from_pretrained(
        args.model_dir,
        num_cls_label=len(labels_to_idx),
        num_tag=len(tags_to_idx),
        ignore_index=tags_to_idx["O"])
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = args.max_steps if args.max_steps > 0 else (
        len(train_data_loader) * args.num_num_train_epochs)
    warmup = args.warmup_steps if args.warmup_steps > 0 else args.warmup_proportion
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         warmup)

    # num_train_optimization_steps = len(train_ds) / args.batch_size / \
    #     args.gradient_accumulation_steps * args.num_train_epochs
    # if args.local_rank != -1:
    #     num_train_optimization_steps = num_train_optimization_steps // paddle.distributed.get_world_size()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{
        "params": [
            p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay) and "crf" not in n
        ],
        "weight_decay": args.weight_decay
    }, {
        "params": [
            p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay) and "crf" not in n
        ],
        "weight_decay": 0.0
    }]
    bert_optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=optimizer_grouped_parameters)
    crf_parameters = [{
        "params": [p for n, p in model.named_parameters() if "crf" in n],
        "weight_decay": args.weight_decay
    }]
    #TODO: check crf lr
    crf_optimizer = paddle.optimizer.AdamW(
        learning_rate=args.learning_rate * 100,
        epsilon=args.adam_epsilon,
        parameters=crf_parameters)

    logger.info("Total steps: %s" % num_training_steps)
    logger.info("WarmUp steps: %s" % warmup)

    # tr_cls_loss = AverageMeter()
    # tr_seq_loss = AverageMeter()
    # tr_seq_acc = AverageMeter()
    # tr_crf_loss = AverageMeter()
    tr_loss = AverageMeter()
    cls_acc = Accuracy(top_k=1)
    seq_acc = SequenceAccuracy()

    global_step = 0

    train_logs = dict()
    for epoch in range(1, args.num_train_epochs + 1):
        logger.info(f"Epoch {epoch} beginnig")
        total_score = 0.0
        # model.train()
        start_time = time.time()

        for total_step, batch in enumerate(train_data_loader):
            global_step += 1

            input_ids, token_type_ids, seq_len, tags, cls_label = batch
            outputs = model(
                input_ids,
                token_type_ids,
                lengths=seq_len,
                tag_labels=tags,
                cls_label=cls_label)
            total_loss, seq_logits, cls_logits = outputs[0], outputs[
                1], outputs[2]
            # total_loss = cls_loss + seq_loss + crf_loss
            # total_score += total_loss.item()
            # if args.n_gpu > 1:
            #     total_loss = total_loss.mean()
            # if args.gradient_accumulation_steps > 1:
            #     total_loss = total_loss / args.gradient_accumulation_steps
            total_loss.backward()
            # crf_loss.backward()

            # tr_cls_loss.update(cls_loss.item(), n=1)
            # tr_seq_loss.update(seq_loss.item(), n=1)
            # tr_crf_loss.update(crf_loss.item(), n=1)
            tr_loss.update(total_loss.numpy(), n=1)
            # seq_acc = SequenceAccuracy()

            # if (total_step + 1) % args.gradient_accumulation_steps == 0:
            bert_optimizer.step()
            crf_optimizer.step()
            bert_optimizer.clear_grad()
            crf_optimizer.clear_grad()
            # model.zero_grad()
            lr_scheduler.step()

            cls_acc(
                logits=cls_logits.view(-1, len(labels_to_idx)),
                target=cls_label.view(-1))
            seq_acc(
                logits=seq_logits.view(-1, len(tags_to_idx)),
                target=tags.view(-1),
                ignore_index=tags_to_idx["O"])

            if global_step % args.logging_steps == 0 and global_step != 0:
                # valid_step = False
                end_time = time.time()
                train_logs["loss"] = tr_loss.avg
                train_logs["Classification Accuracy"] = cls_acc.value()
                train_logs["Sequence Labeling Accuracy"] = seq_acc.value()
                # train_logs["Classification loss"] = tr_cls_loss.avg
                # train_logs["Sequence Labeling loss"] = tr_seq_loss.avg
                # train_logs["CRF likelihood"] = tr_crf_loss.avg
                train_logs["speed"] = (float(args.logging_steps) /
                                       (end_time - start_time))
                logger.info("[Training]["
                            "%s/%s][%s/%s]" % (epoch, args.num_train_epochs,
                                               global_step, num_training_steps)
                            + " - ".join(f"{key}: {value:g} "
                                         for key, value in train_logs.items()))
                start_time = time.time()
                # tr_seq_loss.reset()
                # tr_cls_loss.reset()
                tr_loss.reset()
                # tr_crf_loss.reset()
                cls_acc.reset()
                seq_acc.reset()
                total_score = 0.0

            if (global_step % args.save_steps == 0 or
                    global_step == num_training_steps) and (
                        (not args.n_gpu > 1) or
                        paddle.distributed.get_rank() == 0):
                output_dir = os.path.join(args.output_dir,
                                          "ernie_ctm_ft_model_%d.pdparams" %
                                          (global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # Need better way to get inner model of DataParallel
                model_to_save = model._layers if isinstance(
                    model, paddle.DataParallel) else model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    if args.n_gpu > 1:
        paddle.distributed.spawn(do_train, args=(args, ), nprocs=args.n_gpu)
    else:
        do_train(args)
