from functools import partial
import argparse
import os
import random
import time
import distutils.util

import numpy as np
import paddle
from paddlenlp.data import Pad, Dict
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup, ElectraTokenizer
from paddlenlp.metrics import ChunkEvaluator

from model import ElectraForBinaryTokenClassification
from utils import create_dataloader, convert_example_ner

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu', 'npu'], default='gpu', help='Select which device to train model, default to gpu.')
parser.add_argument('--init_from_ckpt', default=None, type=str, help='The path of checkpoint to be loaded.')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size per GPU/CPU for training.')
parser.add_argument('--learning_rate', default=6e-5, type=float, help='Learning rate for fine-tuning token classification task.')
parser.add_argument('--max_seq_length', default=128, type=int, help='The maximum total input sequence length after tokenization.')
parser.add_argument('--valid_steps', default=100, type=int, help='The interval steps to evaluate model performance.')
parser.add_argument('--logging_steps', default=10, type=int, help='The interval steps to logging.')
parser.add_argument('--save_steps', default=10000, type=int, help='The interval steps to save checkpoints.')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay if we apply some.')
parser.add_argument('--warmup_proportion', default=0.1, type=float, help='Linear warmup proportion over the training process.')
parser.add_argument('--use_amp', default=False, type=bool, help='Enable mixed precision training.')
parser.add_argument('--epochs', default=1, type=int, help='Total number of training epochs.')
parser.add_argument('--seed', default=1000, type=int, help='Random seed.')
parser.add_argument('--save_dir', default='./checkpoint', type=str, help='The output directory where the model checkpoints will be written.')

args = parser.parse_args()
# yapf: enable


def set_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def evaluate(model, criterion, metrics, data_loader):
    model.eval()
    metrics[0].reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, position_ids, masks, label_oth, label_sym = batch
        logits = model(input_ids, token_type_ids, position_ids)
        loss_oth = criterion(logits[0], paddle.unsqueeze(label_oth, 2))
        loss_oth = paddle.mean(loss_oth * paddle.unsqueeze(masks, 2))
        loss_sym = criterion(logits[1], paddle.unsqueeze(label_sym, 2))
        loss_sym = paddle.mean(loss_sym * paddle.unsqueeze(masks, 2))

        losses.append([loss_oth.numpy(), loss_sym.numpy()])

        lengths = paddle.sum(masks, axis=1)
        pred_oth = paddle.argmax(logits[0], axis=2)
        pred_sym = paddle.argmax(logits[1], axis=2)
        correct_oth = metrics[0].compute(lengths, pred_oth, label_oth)
        correct_sym = metrics[1].compute(lengths, pred_sym, label_sym)
        correct_oth = [x.numpy() for x in correct_oth]
        correct_sym = [x.numpy() for x in correct_sym]
        metrics[0].update(*correct_oth)
        metrics[0].update(*correct_sym)
        _, _, result = metrics[0].accumulate()
    loss = np.mean(losses, axis=0)
    print('eval loss symptom: %.5f, loss others: %.5f, f1: %.5f' %
          (loss[1], loss[0], result))
    model.train()
    metrics[0].reset()


def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    train_ds, dev_ds = load_dataset('cblue', 'CMeEE', splits=['train', 'dev'])

    model = ElectraForBinaryTokenClassification.from_pretrained(
        'ehealth-chinese', num_classes=[len(x) for x in train_ds.label_list])
    tokenizer = ElectraTokenizer.from_pretrained('ehealth-chinese')

    label_list = train_ds.label_list
    pad_label_id = [len(label_list[0]) - 1, len(label_list[1]) - 1]
    ignore_label_id = -100

    trans_func = partial(
        convert_example_ner,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        pad_label_id=pad_label_id)

    batchify_fn = lambda samples, fn=Dict({
        'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),
        'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),
        'position_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),
        'attention_mask': Pad(axis=0, pad_val=0, dtype='float32'),
        'label_oth': Pad(axis=0, pad_val=pad_label_id[0], dtype='int64'),
        'label_sym': Pad(axis=0, pad_val=pad_label_id[1], dtype='int64')
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

    criterion = paddle.nn.functional.softmax_with_cross_entropy

    metrics = [ChunkEvaluator(label_list[0]), ChunkEvaluator(label_list[1])]

    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)

    global_step = 0
    tic_train = time.time()
    total_train_time = 0
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids, position_ids, masks, label_oth, label_sym = batch
            with paddle.amp.auto_cast(
                    args.use_amp,
                    custom_white_list=['layer_norm', 'softmax', 'gelu'], ):
                logits = model(input_ids, token_type_ids, position_ids, masks)

                loss_oth = criterion(logits[0], paddle.unsqueeze(label_oth, 2))
                loss_sym = criterion(logits[1], paddle.unsqueeze(label_sym, 2))
                loss_masks = paddle.unsqueeze(masks, 2)
                loss_oth = paddle.mean(loss_oth * loss_masks)
                loss_sym = paddle.mean(loss_sym * loss_masks)

                loss = loss_oth + loss_sym

                lengths = paddle.sum(masks, axis=1)
                pred_oth = paddle.argmax(logits[0], axis=-1)
                pred_sym = paddle.argmax(logits[1], axis=-1)
                correct_oth = metrics[0].compute(lengths, pred_oth, label_oth)
                correct_sym = metrics[1].compute(lengths, pred_sym, label_sym)
                correct_oth = [x.numpy() for x in correct_oth]
                correct_sym = [x.numpy() for x in correct_sym]
                metrics[0].update(*correct_oth)
                metrics[0].update(*correct_sym)
                _, _, f1 = metrics[0].accumulate()

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
                        'global step %d, epoch: %d, batch: %d, loss: %.5f, loss symptom: %.5f, loss others: %.5f, f1: %.5f, speed: %.2f step/s'
                        % (global_step, epoch, step, loss, loss_sym, loss_oth,
                           f1, args.logging_steps / time_diff))
                    tic_train = time.time()

                if global_step % args.valid_steps == 0 and rank == 0:
                    evaluate(model, criterion, metrics, dev_data_loader)
                    tic_train = time.time()

                if global_step % args.save_steps == 0 and rank == 0:
                    save_dir = os.patj.join(args.save_dir,
                                            'model_%d' % global_step)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    if paddle.distributed.get_world_size() > 1:
                        model._layers.save_pretrained(save_dir)
                    else:
                        model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    tic_train = time.time()
    print('Speed: %.2f steps/s' % (global_step / total_train_time))


if __name__ == '__main__':
    do_train()
