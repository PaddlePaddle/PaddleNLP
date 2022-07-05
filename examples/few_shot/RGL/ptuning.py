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
from collections import defaultdict

import paddle
import paddle.nn as nn
from paddle.metric import Accuracy
from paddlenlp.utils.log import logger
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
from visualdl import LogWriter

from template import PTuningTemplate
from verbalizer import ManualVerbalizer
from tokenizer import MLMTokenizerWrapper
from data import load_dataset, InputFeatures
from utils import set_seed, check_args, convert_example, create_dataloader, LinearSchedulerWarmup


def evaluate(model, dataloader, metric, template, verbalizer):
    model.eval()
    metric.reset()
    for batch in tqdm(dataloader):
        batch = template.process_batch(batch)
        logits = model(input_ids=batch['input_ids'],
                       inputs_embeds=batch['input_embeds'],
                       attention_mask=batch['attention_mask'])

        label_logits = verbalizer.process_logits(logits, batch['mask_ids'])
        correct = metric.compute(label_logits, batch['label'])
        metric.update(correct)
    score = metric.accumulate()
    logger.info('Evaluation accuracy: {:.6f}'.format(score))
    model.train()


def contrastive_loss(sentence_embeddings, labels, eq_fn=lambda x, y: x == y):
    """ Compute the loss of RGL method. """
    batch_size = sentence_embeddings.shape[0]
    bce_metric = nn.CrossEntropyLoss()
    cos_metric = nn.CosineSimilarity(axis=0, eps=1e-6)
    loss = 0
    for i in range(batch_size):
        for j in range(batch_size):
            score = cos_metric(sentence_embeddings[i], sentence_embeddings[j])
            score = score.unsqueeze(0)
            logits = paddle.concat([(1 - score) * 50, (1 + score) * 50],
                                   axis=-1)
            if eq_fn(labels[i], labels[j]):
                loss += bce_metric(logits.reshape([-1, logits.shape[-1]]),
                                   paddle.to_tensor([
                                       1,
                                   ]).unsqueeze(0))
            else:
                loss += bce_metric(logits.reshape([-1, logits.shape[-1]]),
                                   paddle.to_tensor([
                                       0,
                                   ]).unsqueeze(0))

    loss = loss / (batch_size * (batch_size - 1))
    loss = loss / 100
    return loss


def main():
    args = parse_arguments()
    paddle.set_device(args.device)
    set_seed(args.seed)

    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer_wrapper = MLMTokenizerWrapper(args.max_seq_length, tokenizer)

    train_ds, dev_ds, test_ds, label_list = load_dataset(
        args.dataset, data_path=args.data_path, splits=['train', 'dev', 'test'])

    template = PTuningTemplate(tokenizer,
                               model,
                               args.template,
                               prompt_encoder='lstm')
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

    if args.freeze_model:
        for param in model.parameters():
            param.stop_gradient = True
        plm_optimizer = None
    else:
        plm_lr_scheduler = LinearSchedulerWarmup(args.learning_rate,
                                                 args.warmup_steps, max_steps)
        decay_params = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ['bias', 'norm'])
        ]
        plm_optimizer = paddle.optimizer.AdamW(
            learning_rate=plm_lr_scheduler,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            grad_clip=paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm),
            apply_decay_param_fun=lambda x: x in decay_params)

    soft_lr_scheduler = LinearSchedulerWarmup(args.soft_learning_rate,
                                              args.soft_warmup_steps, max_steps)
    soft_optimizer = paddle.optimizer.AdamW(
        learning_rate=soft_lr_scheduler,
        parameters=template.parameters(),
        grad_clip=paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm))

    loss_fn = nn.CrossEntropyLoss()
    metric_fn = Accuracy()
    with LogWriter(logdir="./log/ptuning/train") as writer:
        best_metric = -float('inf')
        global_step = 1
        global_loss = 0
        for epoch in range(1, num_epoch + 1):
            for step, batch in enumerate(train_loader, start=1):
                writer.add_scalar('train/lr', plm_lr_scheduler.get_lr(),
                                  global_step)

                batch = template.process_batch(batch)

                logits = model(input_ids=batch['input_ids'],
                               inputs_embeds=batch['input_embeds'],
                               attention_mask=batch['attention_mask'])

                label_logits = verbalizer.process_logits(
                    logits, batch['mask_ids'])

                labels = paddle.to_tensor(batch['label'], dtype='int64')
                loss = loss_fn(label_logits, labels)

                if args.alpha > 0:
                    con_loss = contrastive_loss(logits, labels)
                    loss += args.alpha * con_loss
                global_loss += loss.item()

                writer.add_scalar('train/loss', loss.item(), global_step)

                loss.backward()

                soft_optimizer.step()
                soft_lr_scheduler.step()
                if plm_optimizer:
                    plm_optimizer.step()
                    plm_lr_scheduler.step()
                soft_optimizer.clear_grad()
                if plm_optimizer:
                    plm_optimizer.clear_grad()

                if global_step % args.logging_step == 0:
                    avg_loss = global_loss / args.logging_step
                    logger.info(
                        'Epoch: {:3d}/{:3d}, Global Step: {:4d}, Loss: {:e}'.
                        format(epoch, num_epoch, global_step, avg_loss))
                    global_loss = 0

                if global_step % args.eval_step == 0:
                    evaluate(model, dev_loader, metric_fn, template, verbalizer)

                global_step += 1
                if global_step > max_steps:
                    break

        logger.info('--------------- Test -----------------')
        evaluate(model, test_loader, metric_fn, template, verbalizer)


def parse_arguments():
    parser = argparse.ArgumentParser('Implement of P-tuning paper.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--device',
                        type=str,
                        default='gpu',
                        choices=['gpu', 'cpu'],
                        help='Device to train model, default to gpu.')
    parser.add_argument('--dataset',
                        type=str,
                        default='BoolQ',
                        help='The build-in few-shot dataset.')
    parser.add_argument('--task_name',
                        type=str,
                        default=None,
                        help='The build-in task_name.')
    parser.add_argument('--data_path',
                        type=str,
                        default='./data/k-shot/BoolQ/',
                        help='The path to local dataset in .tsv files.')

    parser.add_argument('--model_name_or_path',
                        type=str,
                        default='albert-xxlarge-v2',
                        help='The build-in pretrained LM or the path '\
                             'to local model parameters.')
    parser.add_argument(
        '--template',
        type=str,
        default=
        "{'text':'text_a'}. {'soft'} Question: {'text':'text_b'}? Answer: {'mask'}.",
        help='The input template.')
    parser.add_argument('--verbalizer',
                        type=str,
                        default="{'False':'No', 'True':'Yes'}",
                        help='The label mapping of output.')
    parser.add_argument('--prompt_encoder_type',
                        type=str,
                        default='lstm',
                        choices=['mlp', 'lstm'],
                        help='The model to encode prompt embeddings.')
    parser.add_argument('--freeze_model',
                        type=bool,
                        default=False,
                        help='Whether to update model parameters.')
    parser.add_argument('--alpha',
                        type=float,
                        default=0,
                        help='The weight of link prediction loss in RGL.')
    parser.add_argument('--max_seq_length',
                        type=int,
                        default=256,
                        help='The maximum length of input text.')
    parser.add_argument('--truncate_mode',
                        type=str,
                        default='tail',
                        choices=['tail', 'head'],
                        help='How to truncate input text.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=1.0,
                        help='The maximum norm of all parameters.')

    parser.add_argument('--num_epoch',
                        type=int,
                        default=3,
                        help='The number of epoch for training.')
    parser.add_argument('--max_steps',
                        type=int,
                        default=250,
                        help='Maximum steps, which overwrites num_epoch.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=2,
                        help='The number of samples used per step.')
    parser.add_argument('--eval_batch_size',
                        type=int,
                        default=8,
                        help='The number of eval samples used per step.')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-5,
                        help='The learning rate of pretrained language model.')
    parser.add_argument('--soft_learning_rate',
                        type=float,
                        default=1e-4,
                        help='The learning rate of soft prompt parameters.')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.1,
                        help='Weight decay for leanring rate of PLM.')
    parser.add_argument('--warmup_steps',
                        type=int,
                        default=0,
                        help='The warmup steps for leanring rate.')
    parser.add_argument('--soft_warmup_steps',
                        type=int,
                        default=0,
                        help='The warmup steps for soft leanring rate.')
    parser.add_argument('--logging_step',
                        type=int,
                        default=50,
                        help='Print logs every logging_step steps.')
    parser.add_argument('--eval_step',
                        type=int,
                        default=20,
                        help='Evaluate model every eval_step steps.')
    parser.add_argument('--output_dir',
                        type=str,
                        default='./ckpt/',
                        help='The path to save checkpoints.')
    parser.add_argument('--overwrite_output',
                        action='store_true',
                        help='Whether overwrite the output_dir.')
    args = parser.parse_args()

    check_args(args)

    for arg in vars(args):
        logger.info(format(arg, '<20') + format(str(getattr(args, arg)), '<'))
    return args


if __name__ == '__main__':
    main()
