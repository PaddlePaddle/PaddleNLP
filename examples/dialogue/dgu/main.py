import os
import random
import time
import numpy as np
from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist
from paddle.io import DataLoader, DistributedBatchSampler, BatchSampler
from paddle.optimizer import AdamW
from paddle.metric import Accuracy

from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import BertTokenizer, BertForSequenceClassification, BertForTokenClassification
from paddlenlp.transformers import LinearDecayWithWarmup

from args import parse_args, set_default_args
import data
import metric

TASK_CLASSES = {
    'udc': (data.UDCv1, metric.RecallAtK),
    'dstc2': (data.DSTC2, metric.JointAccuracy),
    'atis_slot': (data.ATIS_DSF, metric.F1Score),
    'atis_intent': (data.ATIS_DID, Accuracy),
    'mrda': (data.MRDA, Accuracy),
    'swda': (data.SwDA, Accuracy)
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def load_ckpt(args, model, optimizer=None):
    if args.init_from_ckpt:
        params_state_dict = paddle.load(args.init_from_ckpt + '.pdparams')
        model.set_state_dict(params_state_dict)
        if optimizer:
            opt_state_dict = paddle.load(args.init_from_ckpt + '.pdopt')
            optimizer.set_state_dict(opt_state_dict)
        print('Loaded checkpoint from %s' % args.init_from_ckpt)


def save_ckpt(model, optimizer, output_dir, name):
    params_path = os.path.join(output_dir, '{}.pdparams'.format(name))
    opt_path = os.path.join(output_dir, '{}.pdopt'.format(name))
    paddle.save(model.state_dict(), params_path)
    paddle.save(optimizer.state_dict(), opt_path)


class DGULossFunction(nn.Layer):

    def __init__(self, task_name):
        super(DGULossFunction, self).__init__()

        self.task_name = task_name
        self.loss_fn = self.get_loss_fn()

    def get_loss_fn(self):
        if self.task_name in [
                'udc', 'atis_slot', 'atis_intent', 'mrda', 'swda'
        ]:
            return F.cross_entropy
        elif self.task_name == 'dstc2':
            return nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, logits, labels):
        if self.task_name in ['udc', 'atis_intent', 'mrda', 'swda']:
            loss = self.loss_fn(logits, labels)
        elif self.task_name == 'dstc2':
            loss = self.loss_fn(logits, paddle.cast(labels, dtype=logits.dtype))
        elif self.task_name == 'atis_slot':
            labels = paddle.unsqueeze(labels, axis=-1)
            loss = self.loss_fn(logits, labels)
        return loss


def print_logs(args, step, logits, labels, loss, total_time, metric):
    if args.task_name in ['udc', 'atis_intent', 'mrda', 'swda']:
        if args.task_name == 'udc':
            metric = Accuracy()
        metric.reset()
        correct = metric.compute(logits, labels)
        metric.update(correct)
        acc = metric.accumulate()
        print('step %d - loss: %.4f - acc: %.4f - %.3fs/step' %
              (step, loss, acc, total_time / args.logging_steps))
    elif args.task_name == 'dstc2':
        metric.reset()
        metric.update(logits, labels)
        joint_acc = metric.accumulate()
        print('step %d - loss: %.4f - joint_acc: %.4f - %.3fs/step' %
              (step, loss, joint_acc, total_time / args.logging_steps))
    elif args.task_name == 'atis_slot':
        metric.reset()
        metric.update(logits, labels)
        f1_micro = metric.accumulate()
        print('step %d - loss: %.4f - f1_micro: %.4f - %.3fs/step' %
              (step, loss, f1_micro, total_time / args.logging_steps))


def train(args, model, train_data_loader, dev_data_loader, metric, n_procs,
          rank):
    num_examples = len(train_data_loader) * args.batch_size * n_procs
    max_train_steps = args.epochs * len(train_data_loader)
    print("\nNum train examples: %d" % num_examples)
    print("Max train steps: %d" % max_train_steps)
    print("Warmup proportion: %.2f" % args.warmup_proportion)

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, max_train_steps,
                                         args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = AdamW(learning_rate=lr_scheduler,
                      parameters=model.parameters(),
                      weight_decay=args.weight_decay,
                      apply_decay_param_fun=lambda x: x in decay_params,
                      grad_clip=nn.ClipGradByGlobalNorm(args.max_grad_norm))
    loss_fn = DGULossFunction(args.task_name)

    load_ckpt(args, model, optimizer)

    step = 0
    best_metric = 0.0
    total_time = 0.0
    for epoch in range(args.epochs):
        print('\nEpoch %d/%d' % (epoch + 1, args.epochs))
        batch_start_time = time.time()
        for batch in train_data_loader:
            step += 1
            input_ids, segment_ids, labels = batch
            logits = model(input_ids, segment_ids)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            total_time += (time.time() - batch_start_time)
            if step % args.logging_steps == 0:
                print_logs(args, step, logits, labels, loss, total_time, metric)
                total_time = 0.0
            if step % args.save_steps == 0 or step == max_train_steps:
                if rank == 0:
                    save_ckpt(model, optimizer, args.output_dir, step)
                if args.do_eval:
                    print('\nEval begin...')
                    metric_out = evaluation(args, model, dev_data_loader,
                                            metric)
                    if rank == 0 and metric_out > best_metric:
                        best_metric = metric_out
                        save_ckpt(model, optimizer, args.output_dir, 'best')
                        print('Best model, step: %d\n' % step)
            batch_start_time = time.time()


@paddle.no_grad()
def evaluation(args, model, data_loader, metric):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids)
        if args.task_name in ['atis_intent', 'mrda', 'swda']:
            correct = metric.compute(logits, labels)
            metric.update(correct)
        else:
            metric.update(logits, labels)
    model.train()
    metric_out = metric.accumulate()
    print('Total samples: %d' % (len(data_loader) * args.test_batch_size))
    if args.task_name == 'udc':
        print('R1@10: %.4f - R2@10: %.4f - R5@10: %.4f\n' %
              (metric_out[0], metric_out[1], metric_out[2]))
        return metric_out[0]
    elif args.task_name == 'dstc2':
        print('Joint_acc: %.4f\n' % metric_out)
        return metric_out
    elif args.task_name == 'atis_slot':
        print('F1_micro: %.4f\n' % metric_out)
        return metric_out
    elif args.task_name in ['atis_intent', 'mrda', 'swda']:
        print('Acc: %.4f\n' % metric_out)
        return metric_out


def create_data_loader(args, dataset_class, trans_func, batchify_fn, mode):
    dataset = dataset_class(args.data_dir, mode)
    dataset = MapDataset(dataset).map(trans_func, lazy=True)
    if mode == 'train':
        batch_sampler = DistributedBatchSampler(dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True)
    else:
        batch_sampler = BatchSampler(dataset,
                                     batch_size=args.test_batch_size,
                                     shuffle=False)
    data_loader = DataLoader(dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=batchify_fn,
                             return_list=True)
    return data_loader


def main(args):
    paddle.set_device(args.device)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size > 1 and args.do_train:
        dist.init_parallel_env()

    set_seed(args.seed)

    dataset_class, metric_class = TASK_CLASSES[args.task_name]
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    trans_func = partial(dataset_class.convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_len)
    test_trans_func = partial(dataset_class.convert_example,
                              tokenizer=tokenizer,
                              max_seq_length=args.test_max_seq_len)
    metric = metric_class()

    if args.task_name in ('udc', 'dstc2', 'atis_intent', 'mrda', 'swda'):
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
            Stack(dtype='int64')  # label
        ): fn(samples)
        model = BertForSequenceClassification.from_pretrained(
            args.model_name_or_path, num_classes=dataset_class.num_classes())
    elif args.task_name == 'atis_slot':
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
            Pad(axis=0, pad_val=0, dtype='int64')  # label
        ): fn(samples)
        model = BertForTokenClassification.from_pretrained(
            args.model_name_or_path,
            num_classes=dataset_class.num_classes(),
            dropout=0.0)
    if world_size > 1 and args.do_train:
        model = paddle.DataParallel(model)

    if args.do_train:
        train_data_loader = create_data_loader(args, dataset_class, trans_func,
                                               batchify_fn, 'train')
        if args.do_eval:
            dev_data_loader = create_data_loader(args, dataset_class,
                                                 test_trans_func, batchify_fn,
                                                 'dev')
        else:
            dev_data_loader = None
        train(args, model, train_data_loader, dev_data_loader, metric,
              world_size, rank)

    if args.do_test:
        if rank == 0:
            test_data_loader = create_data_loader(args, dataset_class,
                                                  test_trans_func, batchify_fn,
                                                  'test')
            if args.do_train:
                # If do_eval=True, use best model to evaluate the test data.
                # Otherwise, use final model to evaluate the test data.
                if args.do_eval:
                    args.init_from_ckpt = os.path.join(args.output_dir, 'best')
                    load_ckpt(args, model)
            else:
                if not args.init_from_ckpt:
                    raise ValueError('"init_from_ckpt" should be set.')
                load_ckpt(args, model)
            print('\nTest begin...')
            evaluation(args, model, test_data_loader, metric)


def print_args(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == '__main__':
    args = parse_args()
    set_default_args(args)
    print_args(args)

    main(args)
