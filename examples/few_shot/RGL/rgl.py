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
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
from visualdl import LogWriter

from template import ManualTemplate
from verbalizer import ManualVerbalizer
from tokenizer import MLMTokenizerWrapper
from data import load_dataset, InputFeatures
from utils import set_seed, check_args, convert_example, create_dataloader, LinearSchedulerWarmup

METRIC_MAPPING = defaultdict(Accuracy)
METRIC_MAPPING.update({
    'mrpc': AccuracyAndF1(),
    'qqp': AccuracyAndF1(),
    'cola': Mcc(),
    'sts-b': PearsonAndSpearman()
})


def evaluate(model, dataloader, metric, verbalizer=None):
    model.eval()
    metric.reset()
    for batch in dataloader:
        plm_logits, _ = model.roberta(input_ids=batch['input_ids'],
                                      attention_mask=batch['attention_mask'])
        index = [x.squeeze() for x in paddle.where(batch['mask_ids'] == 1)]
        mask_logits = plm_logits[index[0], index[1]]
        logits = model.lm_head(mask_logits)
        label_logits = verbalizer.process_logits(logits)
        if batch['label'].dtype == paddle.float32:
            label_logits = label_logits.reshape([-1, 2])
        correct = metric.compute(label_logits, batch['label'])
        metric.update(correct)
    score = metric.accumulate()
    if isinstance(metric, Accuracy):
        logger.info('Evaluation accuracy: {:.6f}'.format(score))
    elif isinstance(metric, AccuracyAndF1):
        logger.info('Evaluation F1: {:.6f}'.format(score[3]))
    elif isinstance(metric, Mcc):
        logger.info(
            'Evaluation Matthews correlation coefficient: {:.6f}'.format(
                score[0]))
    elif instance(metric, 'PearsonAndSpearman'):
        logger.info('Evaluation Pearson correlation coefficient: {:.6f}'.format(
            score[2]))
    else:
        logger.info('Evaluation score: {}'.format(score))
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

    if args.freeze_model:
        for param in model.parameters():
            param.stop_gradient = True
        optimizer = None
    else:
        lr_scheduler = LinearSchedulerWarmup(args.learning_rate,
                                             args.warmup_steps, max_steps)
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

    loss_fn = nn.CrossEntropyLoss()
    metric_fn = METRIC_MAPPING[args.dataset]
    with LogWriter(logdir="./log/pet/train") as writer:
        best_metric = -float('inf')
        global_step = 1
        for epoch in range(1, num_epoch + 1):
            for step, batch in enumerate(train_loader, start=1):
                writer.add_scalar('train/lr', lr_scheduler.get_lr(),
                                  global_step)

                plm_logits, _ = model.roberta(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'])

                index = [
                    x.squeeze() for x in paddle.where(batch['mask_ids'] == 1)
                ]
                mask_logits = plm_logits[index[0], index[1]]

                logits = model.lm_head(mask_logits)

                label_logits = verbalizer.process_logits(logits)

                if batch['label'].dtype == paddle.float32:
                    loss_fn = nn.KLDivLoss()
                    labels = paddle.stack([
                        1 - batch['label'].reshape([-1]) / 5.,
                        batch['label'].reshape([-1]) / 5.
                    ],
                                          axis=-1)
                    loss = loss_fn(label_logits.reshape([-1, 2]), labels)
                    if args.alpha > 0:
                        con_loss = contrastive_loss(
                            logits, labels, lambda x, y: np.argmax(x, axis=0) ==
                            np.argmax(y, axis=0))
                        loss += args.alpha * con_loss
                else:
                    labels = paddle.to_tensor(batch['label'], dtype='int64')
                    loss = loss_fn(label_logits, labels)
                    if args.alpha > 0:
                        con_loss = contrastive_loss(logits, labels)
                        loss += args.alpha * con_loss

                loss.backward()
                if optimizer:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.clear_grad()

                writer.add_scalar('train/loss', loss.item(), global_step)

                if global_step % args.logging_step == 0:
                    logger.info(
                        'Epoch: {:3d}/{:3d}, Global Step: {:4d}, Loss: {:e}'.
                        format(epoch, num_epoch, global_step, loss.item()))

                if global_step % args.eval_step == 0:
                    evaluate(model, dev_loader, metric_fn, verbalizer)

                global_step += 1
                if global_step > max_steps:
                    break

        logger.info('--------------- Test -----------------')
        evaluate(model, test_loader, metric_fn, verbalizer)


def parse_arguments():
    # yapf: diable
    parser = argparse.ArgumentParser('Implement of RGL paper.')
    parser.add_argument('--seed', type=int, default=1000, help='Random seed.')
    parser.add_argument(
        '--device',
        type=str,
        default='gpu',
        choices=['gpu', 'cpu'],
        help='Select which device to train model, default to gpu.')
    parser.add_argument('--dataset',
                        type=str,
                        default='FewGLUE',
                        help='The build-in few-shot dataset.')
    parser.add_argument('--task_name',
                        type=str,
                        default='sst-2',
                        help='The build-in task_name.')
    parser.add_argument('--data_path',
                        type=str,
                        default=None,
                        help='The path to local dataset in .tsv files.')

    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='roberta-large',
        help='The build-in pretrained LM or the path to local model parameters.'
    )
    parser.add_argument('--use_prompt',
                        type=bool,
                        default=True,
                        help='Whether to use prompt.')
    parser.add_argument('--template',
                        type=str,
                        default="{'text':'text_a'} It was {'mask'}.",
                        help='The input template.')
    parser.add_argument('--verbalizer',
                        type=str,
                        default="{'0':'terrible', '1':'great'}",
                        help='The label mapping of output.')
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
                        default=512,
                        help='The maximum length of input text.')
    parser.add_argument('--truncate_mode',
                        type=str,
                        default='tail',
                        choices=['tail'],
                        help='How to truncate input text.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=1.0,
                        help='The maximum norm of all parameters.')

    parser.add_argument('--num_epoch',
                        type=int,
                        default=0,
                        help='The number of epoch for training.')
    parser.add_argument(
        '--max_steps',
        type=int,
        default=1000,
        help='The maximum steps for training, which overwrites num_epoch.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='The number of samples used per step.')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-5,
                        help='The learning rate of optimizer.')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0,
                        help='Weight decay if we apply some.')
    parser.add_argument('--warmup_steps',
                        type=int,
                        default=0,
                        help='The warmup steps for leanring rate scheduler.')
    parser.add_argument('--logging_step',
                        type=int,
                        default=100,
                        help='Print logs every logging_step steps.')
    parser.add_argument('--eval_step',
                        type=int,
                        default=100,
                        help='Evaluate model every eval_step steps.')
    parser.add_argument('--output_dir',
                        type=str,
                        default='./',
                        help='The path to save checkpoints.')
    parser.add_argument('--overwrite_output',
                        action='store_true',
                        help='Whether overwrite the output_dir.')
    args = parser.parse_args()
    # yapf: enable

    check_args(args)

    for arg in vars(args):
        logger.info(format(arg, '<20') + format(str(getattr(args, arg)), '<'))
    return args


if __name__ == '__main__':
    main()
