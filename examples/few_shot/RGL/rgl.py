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

import os
import argparse
from tqdm import tqdm
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
from paddlenlp.utils.log import logger
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM
from visualdl import LogWriter

from template import ManualTemplate
from verbalizer import ManualVerbalizer
from tokenizer import MLMTokenizerWrapper
from data import load_dataset, InputFeatures, TASK_MAPPING, METRIC_MAPPING
from utils import set_seed, check_args, convert_example, create_dataloader, LinearSchedulerWarmup

# yapf: disable
parser = argparse.ArgumentParser('Implementation of RGL paper.')
parser.add_argument('--seed', type=int, default=1000, help='Random seed.')
parser.add_argument('--device', type=str, default='gpu', choices=['gpu', 'cpu'], help='Device for training, default to gpu.')
parser.add_argument('--dataset', type=str, default='SST-2', help='The build-in few-shot dataset.')
parser.add_argument('--data_path', type=str, default=None, help='The path to local dataset in .tsv files.')

parser.add_argument('--model_name_or_path', type=str, default='roberta-large', help='The build-in pretrained LM or the path to local model parameters.')
parser.add_argument('--template', type=str, default="{'text':'text_a'} It was {'mask'}.", help='The input template.')
parser.add_argument('--verbalizer', type=str, default="{'0':'terrible', '1':'great'}", help='The label mapping of output.')
parser.add_argument('--alpha', type=float, default=0, help='The weight of link prediction loss in RGL.')
parser.add_argument('--max_seq_length', type=int, default=512, help='The maximum length of input text.')
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='The maximum norm of all parameters.')

parser.add_argument('--num_epoch', type=int, default=0, help='The number of epoch for training.')
parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps, which overwrites num_epoch.')
parser.add_argument('--batch_size', type=int, default=32, help='The number of samples used per step.')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='The learning rate of optimizer.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay if we apply some.')
parser.add_argument('--warmup_steps', type=int, default=0, help='The warmup steps for leanring rate scheduler.')
parser.add_argument('--logging_step', type=int, default=100, help='Print logs every logging_step steps.')
parser.add_argument('--eval_step', type=int, default=100, help='Evaluate model every eval_step steps.')
parser.add_argument('--save_best', action='store_true', help= 'Save the best model according to evaluation results. Save the last checkpoint if False.')
parser.add_argument('--output_dir', type=str, default='./checkpoints/', help='The path to save checkpoints.')
parser.add_argument('--overwrite_output', action='store_true', help='Whether overwrite the output_dir.')
args = parser.parse_args()
# yapf: enable

check_args(args)
for arg in vars(args):
    logger.info(format(arg, '<20') + format(str(getattr(args, arg)), '<'))


@paddle.no_grad()
def evaluate(model, dataloader, metric, verbalizer, task_type, bound=(0, 5)):
    if task_type == 'regression':
        logsoftmax = nn.LogSoftmax(axis=-1)
        lb, ub = bound
    model.eval()
    metric.reset()
    logits_list = []
    labels_list = []
    for batch in dataloader:
        logits = model(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask'])
        label_logits = verbalizer.process_logits(logits, batch["mask_ids"])
        if task_type == 'regression':
            label_logits = logsoftmax(label_logits)
            label_logits = paddle.exp(
                label_logits[..., 1].unsqueeze(-1)) * (ub - lb) + lb
        correct = metric.compute(label_logits, batch['label'])
        metric.update(correct)
    score = metric.accumulate()
    score = score if isinstance(score, (list, tuple)) else [score]
    logger.info('{:>20}'.format('Evaluation results:'))
    for name, value in zip(metric.name(), score):
        logger.info('{:>20} = {:.6f}'.format(name, value))
    model.train()
    return score[0]


def contrastive_loss(sentence_embeddings, labels, task_type='classification'):
    """ Compute the loss proposed in RGL method. """

    def _raw_equal(x, y):
        return int(x == y)

    def _max_equal(x, y):
        return int(np.argmax(x, axis=0) == np.argmax(y, axis=0))

    equal_int = _raw_equal if task_type == 'classification' else _max_equal
    bce_metric = nn.CrossEntropyLoss()
    cos_metric = nn.CosineSimilarity(axis=0, eps=1e-6)
    batch_size = sentence_embeddings.shape[0]
    loss = 0
    for i in range(batch_size):
        for j in range(batch_size):
            score = cos_metric(sentence_embeddings[i], sentence_embeddings[j])
            score = score.unsqueeze(0)
            logits = paddle.concat([(1 - score) * 50, (1 + score) * 50],
                                   axis=-1)
            label = paddle.to_tensor(equal_int(labels[i], labels[j]))
            loss += bce_metric(logits.reshape([-1, logits.shape[-1]]),
                               label.unsqueeze(0))
    loss = loss / (batch_size * (batch_size - 1))
    loss = loss / 100
    return loss


def main():
    paddle.set_device(args.device)
    set_seed(args.seed)

    task_type = TASK_MAPPING[args.dataset]
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer_wrapper = MLMTokenizerWrapper(args.max_seq_length, tokenizer)

    train_ds, dev_ds, test_ds, label_list = load_dataset(
        args.dataset, data_path=args.data_path, splits=['train', 'dev', 'test'])

    template = ManualTemplate(tokenizer, args.template)
    logger.info('Set template: {}'.format(template.template))
    verbalizer = ManualVerbalizer(tokenizer,
                                  labels=label_list,
                                  label_to_words=eval(args.verbalizer),
                                  prefix=' ')
    logger.info('Set verbalizer: {}'.format(args.verbalizer))

    trans_fn = partial(convert_example,
                       template=template,
                       verbalizer=verbalizer,
                       tokenizer_wrapper=tokenizer_wrapper)

    train_loader = create_dataloader(train_ds, 'train', args.batch_size,
                                     InputFeatures.collate_fn, trans_fn)
    dev_loader = create_dataloader(dev_ds, 'dev', args.batch_size,
                                   InputFeatures.collate_fn, trans_fn)
    test_loader = create_dataloader(test_ds, 'test', args.batch_size,
                                    InputFeatures.collate_fn, trans_fn)
    if args.max_steps > 0:
        num_epoch = args.max_steps // len(train_loader) + int(
            args.max_steps % len(train_loader) > 0)
        max_steps = args.max_steps
    else:
        num_epoch = args.num_epoch
        max_steps = args.num_epoch * len(train_loader)

    lr_scheduler = LinearSchedulerWarmup(args.learning_rate, args.warmup_steps,
                                         max_steps)
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ['bias', 'norm'])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm),
        apply_decay_param_fun=lambda x: x in decay_params)

    metric_fn = METRIC_MAPPING[args.dataset]
    if task_type == 'regression':
        loss_fn = nn.KLDivLoss()
        lb, ub = 0, 5
        logsoftmax = nn.LogSoftmax(axis=-1)
    else:
        loss_fn = nn.CrossEntropyLoss()
    with LogWriter(logdir="./log/pet/train") as writer:
        best_metric = -float('inf')
        global_step = 1
        global_loss = 0
        for epoch in range(1, num_epoch + 1):
            for step, batch in enumerate(train_loader, start=1):
                writer.add_scalar('train/lr', lr_scheduler.get_lr(),
                                  global_step)

                logits = model(input_ids=batch['input_ids'],
                               attention_mask=batch['attention_mask'])
                label_logits = verbalizer.process_logits(
                    logits, batch["mask_ids"])
                if task_type == 'regression':
                    label_logits = logsoftmax(label_logits)

                    labels = paddle.stack([
                        1 - (batch['label'].reshape([-1]) - lb) / (ub - lb),
                        (batch['label'].reshape([-1]) - lb) / (ub - lb)
                    ],
                                          axis=-1)
                    loss = loss_fn(label_logits.reshape([-1, 2]), labels)
                else:
                    labels = paddle.to_tensor(batch['label'], dtype='int64')
                    loss = loss_fn(
                        label_logits.reshape([-1, label_logits.shape[-1]]),
                        labels.reshape([-1]))
                if args.alpha > 0:
                    con_loss = contrastive_loss(logits,
                                                labels,
                                                task_type=task_type)
                    loss += args.alpha * con_loss
                global_loss += loss.item()

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()

                writer.add_scalar('train/loss', loss.item(), global_step)

                if global_step % args.logging_step == 0:
                    avg_loss = global_loss / args.logging_step
                    logger.info(
                        'Epoch: {:3d}/{:3d}, Global Step: {:4d}, Loss: {:e}'.
                        format(epoch, num_epoch, global_step, avg_loss))
                    global_loss = 0

                if global_step % args.eval_step == 0:
                    logger.info('{0:-^30}'.format(' Validate '))
                    value = evaluate(model, dev_loader, metric_fn, verbalizer,
                                     task_type)
                    if args.save_best and value > best_metric:
                        best_metric = value
                        save_path = os.path.join(args.output_dir, 'model_best')
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        model.save_pretrained(model_path)
                        tokenizer.save_pretrained(model_path)

                global_step += 1
                if global_step > max_steps:
                    break

        logger.info('{0:-^30}'.format(' Test '))
        evaluate(model, test_loader, metric_fn, verbalizer, task_type)
        if not args.save_best:
            save_path = os.path.join(args.output_dir, 'model_last')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)


if __name__ == '__main__':
    main()
