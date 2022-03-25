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

from tqdm import tqdm
import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Dict, Pad, Tuple
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import ElectraTokenizer

from utils import convert_example_spo, create_dataloader, create_batch_label, SPOChunkEvaluator, LinearDecayWithWarmup
from model import ElectraForSPO

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1000, type=int, help='Random seed for initialization.')
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu', 'npu'], default='gpu', help='Select which device to train model, default to gpu.')
parser.add_argument('--epochs', default=100, type=int, help='Total number of training epochs.')
parser.add_argument('--batch_size', default=12, type=int, help='Batch size per GPU/CPU for training.')
parser.add_argument('--learning_rate', default=6e-5, type=float, help='Learning rate for fine-tuning sequence classification task.')
parser.add_argument('--weight_decay', default=0.01, type=float, help="Weight decay of optimizer if we apply some.")
parser.add_argument('--warmup_proportion', default=0.1, type=float, help='Linear warmup proportion of learning rate over the training process.')
parser.add_argument('--max_seq_length', default=300, type=int, help='The maximum total input sequence length after tokenization.')
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
    for batch in tqdm(data_loader):
        input_ids, token_type_ids, position_ids, masks, ent_idx, spo_idx = batch
        max_batch_len = input_ids.shape[-1]
        ent_mask = paddle.unsqueeze(masks, axis=2)
        spo_mask = paddle.matmul(ent_mask, ent_mask, transpose_y=True)
        spo_mask = paddle.unsqueeze(spo_mask, axis=1)

        ent_label, spo_label = create_batch_label(
            ent_idx, spo_idx, metric.num_classes, max_batch_len)

        logits = model(input_ids, token_type_ids, position_ids)
        loss_logits = [F.sigmoid(x) for x in logits]

        ent_loss = criterion(loss_logits[0], ent_label, weight=ent_mask)
        spo_loss = criterion(loss_logits[1], spo_label, weight=spo_mask)
        loss = ent_loss + spo_loss
        losses.append(loss.numpy())
        lengths = paddle.sum(masks, axis=-1)
        correct = metric.compute(lengths, logits[0], logits[1], ent_idx,
                                 spo_idx)
        metric.update(correct)
    results = metric.accumulate()
    print('eval loss: %.5f, entity f1: %.5f, spo f1: %.5f' %
          (np.mean(losses), results['entity'][2], results['spo'][2]))
    model.train()
    metric.reset()


def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    train_ds, dev_ds = load_dataset('cblue', 'CMeIE', splits=['train', 'dev'])

    model = ElectraForSPO.from_pretrained(
        'ehealth-chinese', num_classes=len(train_ds.label_list))
    tokenizer = ElectraTokenizer.from_pretrained('ehealth-chinese')

    trans_func = partial(
        convert_example_spo,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length)

    ignore_pad_id = -1

    batchify_fn = lambda samples, fn=Dict({
        'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),
        'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),
        'position_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),
        'attention_mask': Pad(axis=0, pad_val=0, dtype='float32'),
        'ent_label': lambda x: [x],
        'spo_label': lambda x: [x]
    }): fn(samples)

    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    dev_data_loader = create_dataloader(
        dev_ds,
        mode='dev',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    if args.init_from_ckpt:
        if not os.path.isfile(args.init_from_ckpt):
            raise ValueError('init_from_ckpt is not a valid model filename.')
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = len(train_data_loader) * args.epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ['bias', 'norm'])
    ]

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    criterion = partial(
        paddle.nn.functional.binary_cross_entropy, reduction='sum')

    metric = SPOChunkEvaluator(num_classes=len(train_ds.label_list))

    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
    global_step = 0
    tic_train = time.time()
    total_train_time = 0
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids, position_ids, masks, ent_idx, spo_idx = batch
            max_batch_len = input_ids.shape[-1]
            ent_mask = paddle.unsqueeze(masks, axis=2)
            spo_mask = paddle.matmul(ent_mask, ent_mask, transpose_y=True)
            spo_mask = paddle.unsqueeze(spo_mask, axis=1)

            with paddle.amp.auto_cast(
                    args.use_amp,
                    custom_white_list=['layer_norm', 'softmax', 'gelu'], ):
                logits = model(input_ids, token_type_ids, position_ids)
                ent_label, spo_label = create_batch_label(
                    ent_idx, spo_idx, metric.num_classes, max_batch_len)
                logits = [F.sigmoid(x) for x in logits]
                ent_loss = criterion(logits[0], ent_label, weight=ent_mask)
                spo_loss = criterion(logits[1], spo_label, weight=spo_mask)

                loss = ent_loss + spo_loss

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
                print('global step %d, epoch: %d, batch: %d, loss: %.5f, '
                      'ent_loss: %.5f, spo_loss: %.5f, speed: %.2f steps/s' %
                      (global_step, epoch, step, loss, ent_loss, spo_loss,
                       args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.valid_steps == 0 and rank == 0:
                evaluate(model, criterion, metric, dev_data_loader)
                tic_train = time.time()

            if global_step % args.save_steps == 0 and rank == 0:
                save_dir = os.path.join(args.save_dir, 'model_%d' % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if paddle.distributed.get_world_size() > 1:
                    model._layers.save_pretrained(save_dir)
                else:
                    model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                tic_train = time.time()

    if rank == 0:
        print('Speed: %.2f steps/s' % (global_step / total_train_time))


if __name__ == "__main__":
    do_train()
