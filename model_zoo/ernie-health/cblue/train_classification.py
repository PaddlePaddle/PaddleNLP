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

from functools import partial
import argparse
import os
import random
import time
import distutils.util

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.metric import Accuracy
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import ElectraForSequenceClassification, ElectraTokenizer
from paddlenlp.metrics import MultiLabelsMetric, AccuracyAndF1

from utils import convert_example, create_dataloader, LinearDecayWithWarmup

METRIC_CLASSES = {
    'KUAKE-QIC': Accuracy,
    'KUAKE-QQR': Accuracy,
    'KUAKE-QTR': Accuracy,
    'CHIP-CTC': MultiLabelsMetric,
    'CHIP-STS': MultiLabelsMetric,
    'CHIP-CDN-2C': AccuracyAndF1
}

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['KUAKE-QIC', 'KUAKE-QQR', 'KUAKE-QTR', 'CHIP-STS', 'CHIP-CTC', 'CHIP-CDN-2C'],
                                 default='KUAKE-QIC', type=str, help='Dataset for sequence classfication tasks.')
parser.add_argument('--seed', default=1000, type=int, help='Random seed for initialization.')
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu', 'npu'], default='gpu', help='Select which device to train model, default to gpu.')
parser.add_argument('--epochs', default=3, type=int, help='Total number of training epochs.')
parser.add_argument('--max_steps', default=-1, type=int, help='If > 0: set total number of training steps to perform. Override epochs.')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size per GPU/CPU for training.')
parser.add_argument('--learning_rate', default=6e-5, type=float, help='Learning rate for fine-tuning sequence classification task.')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay of optimizer if we apply some.')
parser.add_argument('--warmup_proportion', default=0.1, type=float, help='Linear warmup proportion of learning rate over the training process.')
parser.add_argument('--max_seq_length', default=128, type=int, help='The maximum total input sequence length after tokenization.')
parser.add_argument('--init_from_ckpt', default=None, type=str, help='The path of checkpoint to be loaded.')
parser.add_argument('--logging_steps', default=10, type=int, help='The interval steps to logging.')
parser.add_argument('--save_dir', default='./checkpoint', type=str, help='The output directory where the model checkpoints will be written.')
parser.add_argument('--save_steps', default=100, type=int, help='The interval steps to save checkpoints.')
parser.add_argument('--valid_steps', default=100, type=int, help='The interval steps to evaluate model performance.')
parser.add_argument('--use_amp', default=False, type=distutils.util.strtobool, help='Enable mixed precision training.')
parser.add_argument('--scale_loss', default=128, type=float, help='The value of scale_loss for fp16.')

args = parser.parse_args()
# yapf: enable


def set_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and compute the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        dataloader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, position_ids, labels = batch
        logits = model(input_ids, token_type_ids, position_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
    if isinstance(metric, Accuracy):
        metric_name = 'accuracy'
        result = metric.accumulate()
    elif isinstance(metric, MultiLabelsMetric):
        metric_name = 'macro f1'
        _, _, result = metric.accumulate('macro')
    else:
        metric_name = 'micro f1'
        _, _, _, result, _ = metric.accumulate()

    print('eval loss: %.5f, %s: %.5f' % (np.mean(losses), metric_name, result))
    model.train()
    metric.reset()


def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    train_ds, dev_ds = load_dataset('cblue',
                                    args.dataset,
                                    splits=['train', 'dev'])

    model = ElectraForSequenceClassification.from_pretrained(
        'ernie-health-chinese',
        num_classes=len(train_ds.label_list),
        activation='tanh')
    tokenizer = ElectraTokenizer.from_pretrained('ernie-health-chinese')

    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'
            ),  # segment
        Pad(axis=0, pad_val=args.max_seq_length - 1, dtype='int64'),  # position
        Stack(dtype='int64')): [data for data in fn(samples)]
    train_data_loader = create_dataloader(train_ds,
                                          mode='train',
                                          batch_size=args.batch_size,
                                          batchify_fn=batchify_fn,
                                          trans_fn=trans_func)
    dev_data_loader = create_dataloader(dev_ds,
                                        mode='dev',
                                        batch_size=args.batch_size,
                                        batchify_fn=batchify_fn,
                                        trans_fn=trans_func)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        state_keys = {
            x: x.replace('discriminator.', '')
            for x in state_dict.keys() if 'discriminator.' in x
        }
        if len(state_keys) > 0:
            state_dict = {
                state_keys[k]: state_dict[k]
                for k in state_keys.keys()
            }
        model.set_dict(state_dict)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = args.max_steps if args.max_steps > 0 else len(
        train_data_loader) * args.epochs
    args.epochs = (num_training_steps - 1) // len(train_data_loader) + 1

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ['bias', 'norm'])
    ]

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    criterion = paddle.nn.loss.CrossEntropyLoss()
    if METRIC_CLASSES[args.dataset] is Accuracy:
        metric = METRIC_CLASSES[args.dataset]()
        metric_name = 'accuracy'
    elif METRIC_CLASSES[args.dataset] is MultiLabelsMetric:
        metric = METRIC_CLASSES[args.dataset](
            num_labels=len(train_ds.label_list))
        metric_name = 'macro f1'
    else:
        metric = METRIC_CLASSES[args.dataset]()
        metric_name = 'micro f1'
    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
    global_step = 0
    tic_train = time.time()
    total_train_time = 0
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids, position_ids, labels = batch
            with paddle.amp.auto_cast(
                    args.use_amp,
                    custom_white_list=['layer_norm', 'softmax', 'gelu', 'tanh'],
            ):
                logits = model(input_ids, token_type_ids, position_ids)
                loss = criterion(logits, labels)
            probs = F.softmax(logits, axis=1)
            correct = metric.compute(probs, labels)
            metric.update(correct)

            if isinstance(metric, Accuracy):
                result = metric.accumulate()
            elif isinstance(metric, MultiLabelsMetric):
                _, _, result = metric.accumulate('macro')
            else:
                _, _, _, result, _ = metric.accumulate()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.minimize(optimizer, loss)
            else:
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            global_step += 1
            if global_step % args.logging_steps == 0 and rank == 0:
                time_diff = time.time() - tic_train
                total_train_time += time_diff
                print(
                    'global step %d, epoch: %d, batch: %d, loss: %.5f, %s: %.5f, speed: %.2f step/s'
                    % (global_step, epoch, step, loss, metric_name, result,
                       args.logging_steps / time_diff))

            if global_step % args.valid_steps == 0 and rank == 0:
                evaluate(model, criterion, metric, dev_data_loader)

            if global_step % args.save_steps == 0 and rank == 0:
                save_dir = os.path.join(args.save_dir, 'model_%d' % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if paddle.distributed.get_world_size() > 1:
                    model._layers.save_pretrained(save_dir)
                else:
                    model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)

            if global_step >= num_training_steps:
                return
            tic_train = time.time()

    if rank == 0 and total_train_time > 0:
        print('Speed: %.2f steps/s' % (global_step / total_train_time))


if __name__ == "__main__":
    do_train()
