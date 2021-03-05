# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
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
import json
from functools import partial
import codecs
import zipfile
import re
from tqdm import tqdm
import sys

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader

from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification, LinearDecayWithWarmup

from data_loader import DuIEDataset, DataCollator
from utils import decoding, find_entity, get_precision_recall_f1, write_prediction_results

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--do_train", action='store_true', default=False, help="do train")
parser.add_argument("--do_predict", action='store_true', default=False, help="do predict")
parser.add_argument("--init_checkpoint", default=None, type=str, required=False, help="Path to initialize params from")
parser.add_argument("--data_path", default="./data", type=str, required=False, help="Path to data.")
parser.add_argument("--predict_data_file", default="./data/test_data.json", type=str, required=False, help="Path to data.")
parser.add_argument("--output_dir", default="./checkpoints", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int,help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--seed", default=42, type=int, help="random seed for initialization")
parser.add_argument("--n_gpu", default=1, type=int, help="number of gpus to use, 0 for cpu.")
args = parser.parse_args()
# yapf: enable


class BCELossForDuIE(nn.Layer):
    def __init__(self, ):
        super(BCELossForDuIE, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, mask):
        loss = self.criterion(logits, labels)
        mask = paddle.cast(mask, 'float32')
        loss = loss * mask.unsqueeze(-1)
        loss = paddle.sum(loss.mean(axis=2), axis=1) / paddle.sum(mask, axis=1)
        loss = loss.mean()
        return loss


def set_random_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def evaluate(model, criterion, data_loader, file_path, mode):
    """
    mode eval:
    eval on development set and compute P/R/F1, called between training.
    mode predict:
    eval on development / test set, then write predictions to \
        predict_test.json and predict_test.json.zip \
        under args.data_path dir for later submission or evaluation.
    """
    model.eval()
    probs_all = None
    seq_len_all = None
    tok_to_orig_start_index_all = None
    tok_to_orig_end_index_all = None
    loss_all = 0
    eval_steps = 0
    for batch in tqdm(data_loader, total=len(data_loader)):
        eval_steps += 1
        input_ids, seq_len, tok_to_orig_start_index, tok_to_orig_end_index, labels = batch
        logits = model(input_ids=input_ids)
        mask = (input_ids != 0).logical_and((input_ids != 1)).logical_and(
            (input_ids != 2))
        loss = criterion(logits, labels, mask)
        loss_all += loss.numpy().item()
        probs = F.sigmoid(logits)
        if probs_all is None:
            probs_all = probs.numpy()
            seq_len_all = seq_len.numpy()
            tok_to_orig_start_index_all = tok_to_orig_start_index.numpy()
            tok_to_orig_end_index_all = tok_to_orig_end_index.numpy()
        else:
            probs_all = np.append(probs_all, probs.numpy(), axis=0)
            seq_len_all = np.append(seq_len_all, seq_len.numpy(), axis=0)
            tok_to_orig_start_index_all = np.append(
                tok_to_orig_start_index_all,
                tok_to_orig_start_index.numpy(),
                axis=0)
            tok_to_orig_end_index_all = np.append(
                tok_to_orig_end_index_all,
                tok_to_orig_end_index.numpy(),
                axis=0)
    loss_avg = loss_all / eval_steps
    print("eval loss: %f" % (loss_avg))

    id2spo_path = os.path.join(os.path.dirname(file_path), "id2spo.json")
    with open(id2spo_path, 'r', encoding='utf8') as fp:
        id2spo = json.load(fp)
    formatted_outputs = decoding(file_path, id2spo, probs_all, seq_len_all,
                                 tok_to_orig_start_index_all,
                                 tok_to_orig_end_index_all)
    if mode == "predict":
        predict_file_path = os.path.join(args.data_path, 'predictions.json')
    else:
        predict_file_path = os.path.join(args.data_path, 'predict_eval.json')

    predict_zipfile_path = write_prediction_results(formatted_outputs,
                                                    predict_file_path)

    if mode == "eval":
        precision, recall, f1 = get_precision_recall_f1(file_path,
                                                        predict_zipfile_path)
        os.system('rm {} {}'.format(predict_file_path, predict_zipfile_path))
        return precision, recall, f1
    elif mode != "predict":
        raise Exception("wrong mode for eval func")


def do_train():
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    # Reads label_map.
    label_map_path = os.path.join(args.data_path, "predicate2id.json")
    if not (os.path.exists(label_map_path) and os.path.isfile(label_map_path)):
        sys.exit("{} dose not exists or is not a file.".format(label_map_path))
    with open(label_map_path, 'r', encoding='utf8') as fp:
        label_map = json.load(fp)
    num_classes = (len(label_map.keys()) - 2) * 2 + 2

    # Loads pretrained model ERNIE
    model = ErnieForTokenClassification.from_pretrained(
        "ernie-1.0", num_classes=num_classes)
    model = paddle.DataParallel(model)
    tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    criterion = BCELossForDuIE()

    # Loads dataset.
    train_dataset = DuIEDataset.from_file(
        os.path.join(args.data_path, 'train_data.json'), tokenizer,
        args.max_seq_length, True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    collator = DataCollator()
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=collator,
        return_list=True)
    eval_file_path = os.path.join(args.data_path, 'dev_data.json')
    test_dataset = DuIEDataset.from_file(eval_file_path, tokenizer,
                                         args.max_seq_length, True)
    test_batch_sampler = paddle.io.BatchSampler(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_sampler=test_batch_sampler,
        collate_fn=collator,
        return_list=True)

    # Defines learning rate strategy.
    steps_by_epoch = len(train_data_loader)
    num_training_steps = steps_by_epoch * args.num_train_epochs
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_ratio)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])])

    # Starts training.
    global_step = 0
    logging_steps = 50
    save_steps = 10000
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        print("\n=====start training of %d epochs=====" % epoch)
        tic_epoch = time.time()
        model.train()
        for step, batch in enumerate(train_data_loader):
            input_ids, seq_lens, tok_to_orig_start_index, tok_to_orig_end_index, labels = batch
            logits = model(input_ids=input_ids)
            mask = (input_ids != 0).logical_and((input_ids != 1)).logical_and(
                (input_ids != 2))
            loss = criterion(logits, labels, mask)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            loss_item = loss.numpy().item()

            if global_step % logging_steps == 0 and paddle.distributed.get_rank(
            ) == 0:
                print(
                    "epoch: %d / %d, steps: %d / %d, loss: %f, speed: %.2f step/s"
                    % (epoch, args.num_train_epochs, step, steps_by_epoch,
                       loss_item, logging_steps / (time.time() - tic_train)))
                tic_train = time.time()

            if global_step % save_steps == 0 and global_step != 0 and paddle.distributed.get_rank(
            ) == 0:
                print("\n=====start evaluating ckpt of %d steps=====" %
                      global_step)
                precision, recall, f1 = evaluate(
                    model, criterion, test_data_loader, eval_file_path, "eval")
                print("precision: %.2f\t recall: %.2f\t f1: %.2f\t" %
                      (100 * precision, 100 * recall, 100 * f1))
                if (not args.n_gpu > 1) or paddle.distributed.get_rank() == 0:
                    print("saving checkpoing model_%d.pdparams to %s " %
                          (global_step, args.output_dir))
                    paddle.save(model.state_dict(),
                                os.path.join(args.output_dir,
                                             "model_%d.pdparams" % global_step))
                model.train()  # back to train mode

            global_step += 1
        tic_epoch = time.time() - tic_epoch
        print("epoch time footprint: %d hour %d min %d sec" %
              (tic_epoch // 3600, (tic_epoch % 3600) // 60, tic_epoch % 60))

    # Does final evaluation.
    if paddle.distributed.get_rank() == 0:
        print("\n=====start evaluating last ckpt of %d steps=====" %
              global_step)
        precision, recall, f1 = evaluate(model, criterion, test_data_loader,
                                         eval_file_path, "eval")
        print("precision: %.2f\t recall: %.2f\t f1: %.2f\t" %
              (100 * precision, 100 * recall, 100 * f1))
        if (not args.n_gpu > 1) or paddle.distributed.get_rank() == 0:
            paddle.save(model.state_dict(),
                        os.path.join(args.output_dir,
                                     "model_%d.pdparams" % global_step))
        print("\n=====training complete=====")


def do_predict():
    # Reads label_map.
    label_map_path = os.path.join(args.data_path, "predicate2id.json")
    if not (os.path.exists(label_map_path) and os.path.isfile(label_map_path)):
        sys.exit("{} dose not exists or is not a file.".format(label_map_path))
    with open(label_map_path, 'r', encoding='utf8') as fp:
        label_map = json.load(fp)
    num_classes = (len(label_map.keys()) - 2) * 2 + 2

    # Loads pretrained model ERNIE
    model = ErnieForTokenClassification.from_pretrained(
        "ernie-1.0", num_classes=num_classes)
    tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    criterion = BCELossForDuIE()

    # Loads dataset.
    test_dataset = DuIEDataset.from_file(args.predict_data_file, tokenizer,
                                         args.max_seq_length, True)
    collator = DataCollator()
    test_batch_sampler = paddle.io.BatchSampler(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_sampler=test_batch_sampler,
        collate_fn=collator,
        return_list=True)

    # Loads model parameters.
    if not (os.path.exists(args.init_checkpoint) and
            os.path.isfile(args.init_checkpoint)):
        sys.exit("wrong directory: init checkpoints {} not exist".format(
            args.init_checkpoint))
    state_dict = paddle.load(args.init_checkpoint)
    model.set_dict(state_dict)

    # Does predictions.
    print("\n=====start predicting=====")
    evaluate(model, criterion, test_data_loader, args.predict_data_file,
             "predict")
    print("=====predicting complete=====")


if __name__ == "__main__":
    set_random_seed(args.seed)
    paddle.set_device("gpu" if args.n_gpu else "cpu")

    if args.do_train:
        if args.n_gpu > 1:
            paddle.distributed.spawn(do_train, nprocs=args.n_gpu)
        else:
            do_train()
    if args.do_predict:
        do_predict()
