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

from utils import convert_example_spo, create_dataloader, SPOChunkEvaluator, LinearDecayWithWarmup
from model import ElectraForSPO

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1000, type=int, help='Random seed for initialization.')
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu', 'npu'], default='gpu', help='Select which device to train model, default to gpu.')
parser.add_argument('--epochs', default=100, type=int, help='Total number of training epochs.')
parser.add_argument('--max_steps', default=-1, type=int, help='If > 0: set total number of training steps to perform. Override epochs.')
parser.add_argument('--batch_size', default=12, type=int, help='Batch size per GPU/CPU for training.')
parser.add_argument('--learning_rate', default=6e-5, type=float, help='Learning rate for fine-tuning sequence classification task.')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay of optimizer if we apply some.')
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
        criterion(`paddle.nn.functional`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    for batch in tqdm(data_loader):
        input_ids, token_type_ids, position_ids, masks, ent_label, spo_label = batch
        max_batch_len = input_ids.shape[-1]
        ent_mask = paddle.unsqueeze(masks, axis=2)
        spo_mask = paddle.matmul(ent_mask, ent_mask, transpose_y=True)
        spo_mask = paddle.unsqueeze(spo_mask, axis=1)

        logits = model(input_ids, token_type_ids, position_ids)

        ent_loss = criterion(logits[0],
                             ent_label[0],
                             weight=ent_mask,
                             reduction='sum')
        spo_loss = criterion(logits[1],
                             spo_label[0],
                             weight=spo_mask,
                             reduction='sum')
        loss = ent_loss + spo_loss
        losses.append(loss.numpy())
        lengths = paddle.sum(masks, axis=-1)
        correct = metric.compute(lengths, logits[0], logits[1], ent_label[1],
                                 spo_label[1])
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

    model = ElectraForSPO.from_pretrained('ernie-health-chinese',
                                          num_classes=len(train_ds.label_list))
    tokenizer = ElectraTokenizer.from_pretrained('ernie-health-chinese')

    trans_func = partial(convert_example_spo,
                         tokenizer=tokenizer,
                         num_classes=len(train_ds.label_list),
                         max_seq_length=args.max_seq_length)

    def batchify_fn(data):
        _batchify_fn = lambda samples, fn=Dict({
            'input_ids':
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),
            'token_type_ids':
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),
            'position_ids':
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),
            'attention_mask':
            Pad(axis=0, pad_val=0, dtype='float32'),
        }): fn(samples)
        ent_label = [x['ent_label'] for x in data]
        spo_label = [x['spo_label'] for x in data]
        input_ids, token_type_ids, position_ids, masks = _batchify_fn(data)
        batch_size, batch_len = input_ids.shape
        num_classes = len(train_ds.label_list)
        # Create one-hot labels.
        #
        # For example,
        # - text:
        #   [CLS], 局, 部, 皮, 肤, 感, 染, 引, 起, 的, 皮, 疹, 等, [SEP]
        #
        # - ent_label (obj: `list`):
        #   [(0, 5), (9, 10)] # ['局部皮肤感染', '皮疹']
        #
        # - one_hot_ent_label: # shape (sequence_length, 2)
        #   [[ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0], # start index
        #    [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0]] # end index
        #
        # - spo_label (obj: `list`):
        #   [(0, 23, 9)] # [('局部皮肤感染', '相关（导致）', '皮疹')], where entities
        #                  are encoded by their start indexes.
        #
        # - one_hot_spo_label: # shape (num_predicate, sequence_length, sequence_length)
        #   [...,
        #    [..., [0, ..., 1, ..., 0], ...], # for predicate '相关（导致）'
        #    ...]                             # the value at [23, 1, 10] is set as 1
        #
        one_hot_ent_label = np.zeros([batch_size, batch_len, 2],
                                     dtype=np.float32)
        one_hot_spo_label = np.zeros(
            [batch_size, num_classes, batch_len, batch_len], dtype=np.float32)
        for idx, ent_idxs in enumerate(ent_label):
            # Shift index by 1 because input_ids start with [CLS] here.
            for x, y in ent_idxs:
                x = x + 1
                y = y + 1
                if x > 0 and x < batch_len and y < batch_len:
                    one_hot_ent_label[idx, x, 0] = 1
                    one_hot_ent_label[idx, y, 1] = 1
        for idx, spo_idxs in enumerate(spo_label):
            for s, p, o in spo_idxs:
                s_id = s[0] + 1
                o_id = o[0] + 1
                if s_id > 0 and s_id < batch_len and o_id < batch_len:
                    one_hot_spo_label[idx, p, s_id, o_id] = 1
        # one_hot_xxx_label are used for loss computation.
        # xxx_label are used for metric computation.
        ent_label = [one_hot_ent_label, ent_label]
        spo_label = [one_hot_spo_label, spo_label]
        return input_ids, token_type_ids, position_ids, masks, ent_label, spo_label

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

    if args.init_from_ckpt:
        if not os.path.isfile(args.init_from_ckpt):
            raise ValueError('init_from_ckpt is not a valid model filename.')
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
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ['bias', 'norm'])
    ]

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    criterion = F.binary_cross_entropy_with_logits

    metric = SPOChunkEvaluator(num_classes=len(train_ds.label_list))

    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
    global_step = 0
    tic_train = time.time()
    total_train_time = 0
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids, position_ids, masks, ent_label, spo_label = batch
            max_batch_len = input_ids.shape[-1]
            ent_mask = paddle.unsqueeze(masks, axis=2)
            spo_mask = paddle.matmul(ent_mask, ent_mask, transpose_y=True)
            spo_mask = paddle.unsqueeze(spo_mask, axis=1)

            with paddle.amp.auto_cast(
                    args.use_amp,
                    custom_white_list=['layer_norm', 'softmax', 'gelu'],
            ):
                logits = model(input_ids, token_type_ids, position_ids)
                ent_loss = criterion(logits[0],
                                     ent_label[0],
                                     weight=ent_mask,
                                     reduction='sum')
                spo_loss = criterion(logits[1],
                                     spo_label[0],
                                     weight=spo_mask,
                                     reduction='sum')

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
