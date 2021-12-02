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

import argparse
import logging
import os
import random
import time
import math
import json
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader
from paddle.metric import Accuracy, Precision, Recall
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import BertModel, BertForTokenClassification, BertTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.log import logger
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
from paddleslim.nas.ofa import OFA, DistillConfig, utils
from paddleslim.nas.ofa.utils import nlp_utils
from paddleslim.nas.ofa.convert_super import Convert, supernet

METRIC_CLASSES = {
    "msra": ChunkEvaluator
}

MODEL_CLASSES = {"bert": (BertForTokenClassification, BertTokenizer), }


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()), )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--lambda_logit",
        default=1.0,
        type=float,
        help="lambda for logit loss.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--seed", type=int, default=1000, help="random seed for initialization")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["gpu", "cpu", "xpu"],
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
    parser.add_argument(
        '--width_mult_list',
        nargs='+',
        type=float,
        default=[1.0, 5 / 6, 2 / 3, 0.5],
        help="width mult in compress")
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


@paddle.no_grad()
def evaluate(epoch, global_step, model, loss_fct, metric, data_loader, width_mult=1.0):
    model.eval()
    metric.reset()
    losses = []

    for batch in data_loader:
        input_ids, token_type_ids, length, labels = batch
        logits = model(input_ids, token_type_ids, attention_mask=[None, None])
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = loss_fct(logits, labels)
        avg_loss = paddle.mean(loss)
        losses.append(avg_loss)
        preds = logits.argmax(axis=2)
        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(
            length, preds, labels)
        metric.update(num_infer_chunks.numpy(),
                      num_label_chunks.numpy(), num_correct_chunks.numpy())
    res = metric.accumulate()
    eval_loss = np.mean(losses)

    # Teacher model's evaluation
    if width_mult == 100:
        if isinstance(metric, ChunkEvaluator):
            print(
                "teacher model, eval loss: %f, precision: %s, recall: %s, f1: %s"
                % (
                    eval_loss,
                    res[0],
                    res[1],
                    res[2],
                   ),
                end='')
            with open(os.path.join(args.output_dir, "train_result.txt"), 'a', encoding='utf-8') as fp:
                fp.write(
                    'epoch:%s, step: %s, teacher model, eval loss: %f, precision: %s, recall: %s, f1: %s\n'
                    % (
                        epoch,
                        global_step,
                        eval_loss,
                        res[0],
                        res[1],
                        res[2],
                       ))

        elif isinstance(metric, AccuracyAndF1):
            print(
                "teacher model, eval loss: %f, acc: %s, precision: %s, recall: %s, f1: %s, acc and f1: %s, "
                % (
                    eval_loss,
                    res[0],
                    res[1],
                    res[2],
                    res[3],
                    res[4], ),
                end='')
            with open(os.path.join(args.output_dir, "train_result.txt"), 'a', encoding='utf-8') as fp:
                fp.write(
                    'epoch:%s, step: %s, teacher model, eval loss: %f, acc: %s,'
                    ' precision: %s, recall: %s, f1: %s, acc and f1: %s, \n'
                    % (
                        epoch,
                        global_step,
                        eval_loss,
                        res[0],
                        res[1],
                        res[2],
                        res[3],
                        res[4],))
        elif isinstance(metric, Mcc):
            print(
                "teacher model, eval loss: %f, mcc: %s, " %
                (eval_loss, res[0]),
                end='')
            with open(os.path.join(args.output_dir, "train_result.txt"), 'a', encoding='utf-8') as fp:
                fp.write(
                    'epoch:%s, step: %s, teacher model, eval loss: %f, mcc: %s, \n'
                    % (epoch, global_step, eval_loss, res[0]))
        elif isinstance(metric, PearsonAndSpearman):
            print(
                "teacher model, eval loss: %f, pearson: %s, spearman: %s, pearson and spearman: %s, "
                % (eval_loss, res[0], res[1], res[2]),
                end='')
            with open(os.path.join(args.output_dir, "train_result.txt"), 'a', encoding='utf-8') as fp:
                fp.write(
                    'epoch:%s, step: %s, teacher model, eval loss: %f, '
                    'pearson: %s, spearman: %s, pearson and spearman: %s, \n'
                    % (epoch, global_step, eval_loss, res[0], res[1], res[2]))
        else:
            print(
                "teacher model, eval loss: %f, acc: %s, " % (eval_loss, res),
                end='')
            with open(os.path.join(args.output_dir, "train_result.txt"), 'a', encoding='utf-8') as fp:
                fp.write(
                    'epoch:%s, step: %s, teacher model, eval loss: %f, acc: %s,  \n'
                    % (epoch, global_step, eval_loss, res))
    else:
        if isinstance(metric, ChunkEvaluator):
            print(
                "width_mult: %s, eval loss: %f, precision: %s, recall: %s, f1: %s,"
                % (
                    width_mult,
                    eval_loss,
                    res[0],
                    res[1],
                    res[2],
                  ),
                end='')
            with open(os.path.join(args.output_dir, "train_result.txt"), 'a', encoding='utf-8') as fp:
                fp.write(
                    'epoch:%s, step: %s, width_mult: %s, eval loss: %f, precision: %s, recall: %s, f1: %s\n'
                    % (
                        epoch,
                        global_step,
                        width_mult,
                        eval_loss,
                        res[0],
                        res[1],
                        res[2],
                      ))

        elif isinstance(metric, AccuracyAndF1):
            print(
                "width_mult: %s, eval loss: %f, acc: %s, precision: %s, recall: %s, f1: %s, acc and f1: %s, "
                % (
                    width_mult,
                    eval_loss,
                    res[0],
                    res[1],
                    res[2],
                    res[3],
                    res[4], ),
                end='')
            with open(os.path.join(args.output_dir, "train_result.txt"), 'a', encoding='utf-8') as fp:
                fp.write(
                    'epoch:%s, step: %s, width_mult: %s, eval loss: %f, acc: %s, '
                    'precision: %s, recall: %s, f1: %s, acc and f1: %s,  \n'
                    % (
                        epoch,
                        global_step,
                        width_mult,
                        eval_loss,
                        res[0],
                        res[1],
                        res[2],
                        res[3],
                        res[4],))
        elif isinstance(metric, Mcc):
            print(
                "width_mult: %s, eval loss: %f, mcc: %s, " %
                (str(width_mult), eval_loss, res[0]),
                end='')
            with open(os.path.join(args.output_dir, "train_result.txt"), 'a', encoding='utf-8') as fp:
                fp.write(
                    'epoch:%s, step: %s, width_mult: %s, eval loss: %f, mcc: %s, \n'
                    % (epoch, global_step, str(width_mult), eval_loss, res[0]))
        elif isinstance(metric, PearsonAndSpearman):
            print(
                "width_mult: %s, eval loss: %f, pearson: %s, spearman: %s, pearson and spearman: %s, "
                % (str(width_mult), eval_loss, res[0], res[1], res[2]),
                end='')
            with open(os.path.join(args.output_dir, "train_result.txt"), 'a', encoding='utf-8') as fp:
                fp.write(
                    'epoch:%s, step: %s, width_mult: %s, eval loss: %f, pearson: %s, '
                    'spearman: %s, pearson and spearman: %s, \n'
                    % (epoch, global_step, str(width_mult), eval_loss, res[0], res[1], res[2]))
        else:
            print(
                "width_mult: %s, eval loss: %f, acc: %s, " %
                (str(width_mult), eval_loss, res),
                end='')
            with open(os.path.join(args.output_dir, "train_result.txt"), 'a', encoding='utf-8') as fp:
                fp.write(
                    'epoch:%s, step: %s, width_mult: %s, eval loss: %f, acc: %s,  \n'
                    % (epoch, global_step, str(width_mult), eval_loss, res))
    model.train()
    return res, eval_loss


# monkey patch for bert forward to accept [attention_mask, head_mask] as  attention_mask
def bert_forward(self,
                 input_ids,
                 token_type_ids=None,
                 position_ids=None,
                 attention_mask=[None, None]):
    wtype = self.pooler.dense.fn.weight.dtype if hasattr(
        self.pooler.dense, 'fn') else self.pooler.dense.weight.dtype
    if attention_mask[0] is None:
        attention_mask[0] = paddle.unsqueeze(
            (input_ids == self.pad_token_id).astype(wtype) * -1e9, axis=[1, 2])
    embedding_output = self.embeddings(
        input_ids=input_ids,
        position_ids=position_ids,
        token_type_ids=token_type_ids)
    encoder_outputs = self.encoder(embedding_output, attention_mask)
    sequence_output = encoder_outputs
    pooled_output = self.pooler(sequence_output)
    return sequence_output, pooled_output


BertModel.forward = bert_forward


# reorder weights according head importance and neuron importance
def reorder_neuron_head(model, head_importance, neuron_importance):
    # reorder heads and ffn neurons
    for layer, current_importance in enumerate(neuron_importance):
        # reorder heads
        idx = paddle.argsort(head_importance[layer], descending=True)
        nlp_utils.reorder_head(model.bert.encoder.layers[layer].self_attn, idx)
        # reorder neurons
        idx = paddle.argsort(
            paddle.to_tensor(current_importance), descending=True)
        nlp_utils.reorder_neuron(
            model.bert.encoder.layers[layer].linear1.fn, idx, dim=1)
        nlp_utils.reorder_neuron(
            model.bert.encoder.layers[layer].linear2.fn, idx, dim=0)


def soft_cross_entropy(inp, target):
    inp_likelihood = F.log_softmax(inp, axis=-1)
    target_prob = F.softmax(target, axis=-1)
    return -1. * paddle.mean(paddle.sum(inp_likelihood * target_prob, axis=-1))


def tokenize_and_align_labels(example, tokenizer, no_entity_id,
                              max_seq_len=512):
    labels = example['labels']
    example = example['tokens']
    tokenized_input = tokenizer(
        example,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)

    # -2 for [CLS] and [SEP]
    if len(tokenized_input['input_ids']) - 2 < len(labels):
        labels = labels[:len(tokenized_input['input_ids']) - 2]
    tokenized_input['labels'] = [no_entity_id] + labels + [no_entity_id]
    tokenized_input['labels'] += [no_entity_id] * (
        len(tokenized_input['input_ids']) - len(tokenized_input['labels']))
    return tokenized_input


def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)

    args.task_name = args.task_name.lower()
    metric_class = METRIC_CLASSES[args.task_name]
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Create dataset, tokenizer and dataloader.
    train_ds, dev_ds = load_dataset(
        'msra_ner', splits=('train', 'test'), lazy=False)
    label_list = train_ds.label_list
    label_num = len(label_list)
    no_entity_id = label_num - 1

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    trans_func = partial(
        tokenize_and_align_labels,
        tokenizer=tokenizer,
        no_entity_id=no_entity_id,
        max_seq_len=args.max_seq_length)

    train_ds = train_ds.map(trans_func, lazy=True)

    ignore_label = -100

    batchify_fn = lambda samples, fn=Dict({
        'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int32'),  # input
        'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int32'),  # segment
        'seq_len': Stack(dtype='int64'),  # seq_len
        'labels': Pad(axis=0, pad_val=ignore_label, dtype='int64')  # label
    }): fn(samples)

    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    train_data_loader = DataLoader(
        dataset=train_ds,
        collate_fn=batchify_fn,
        num_workers=0,
        batch_sampler=train_batch_sampler,
        return_list=True)

    dev_ds = dev_ds.map(trans_func, lazy=True)
    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=args.batch_size, shuffle=False)
    dev_data_loader = DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    batchify_fn_d = lambda samples, fn=Dict({
        'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int32'),  # input
        'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int32'),  # segment
        'labels': Pad(axis=0, pad_val=ignore_label, dtype='int64')  # label
    }): fn(samples)

    dev_data_loader_d = DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn_d,
        num_workers=0,
        return_list=True)

    model = model_class.from_pretrained(
        args.model_name_or_path, num_classes=label_num)

    # Step1: Initialize a dictionary to save the weights from the origin BERT model.
    origin_weights = model.state_dict()

    # Step2: Convert origin model to supernet.
    sp_config = supernet(expand_ratio=args.width_mult_list)
    model = Convert(sp_config).convert(model)
    # Use weights saved in the dictionary to initialize supernet.
    utils.set_state_dict(model, origin_weights)
    del origin_weights

    # Step3: Define teacher model.
    teacher_model = model_class.from_pretrained(
        args.model_name_or_path, num_classes=label_num)

    # Step4: Config about distillation.
    mapping_layers = ['bert.embeddings']
    for idx in range(model.bert.config['num_hidden_layers']):
        mapping_layers.append('bert.encoder.layers.{}'.format(idx))

    default_distill_config = {
        'lambda_distill': 0.1,
        'teacher_model': teacher_model,
        'mapping_layers': mapping_layers,
    }
    distill_config = DistillConfig(**default_distill_config)

    # Step5: Config in supernet training.
    ofa_model = OFA(model,
                    distill_config=distill_config,
                    elastic_order=['width'])

    criterion = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)

    metric = metric_class(label_list=label_list)

    # Step6: Calculate the importance of neurons and head,
    # and then reorder them according to the importance.
    head_importance, neuron_importance = nlp_utils.compute_neuron_head_importance(
        args.task_name,
        ofa_model.model,
        dev_data_loader_d,
        loss_fct=criterion,
        num_layers=model.bert.config['num_hidden_layers'],
        num_heads=model.bert.config['num_attention_heads'])
    reorder_neuron_head(ofa_model.model, head_importance, neuron_importance)

    if paddle.distributed.get_world_size() > 1:
        ofa_model.model = paddle.DataParallel(ofa_model.model)

    if args.max_steps > 0:
        num_training_steps = args.max_steps
        num_train_epochs = math.ceil(num_training_steps /
                                     len(train_data_loader))
    else:
        num_training_steps = len(train_data_loader) * args.num_train_epochs
        num_train_epochs = args.num_train_epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_steps)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=ofa_model.model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    best_f1 = {}.fromkeys(range(4), 0.0)
    global_step = 0
    tic_train = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for epoch in range(num_train_epochs):
        # Step7: Set current epoch and task.
        ofa_model.set_epoch(epoch)
        ofa_model.set_task('width')

        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, segment_ids, _, labels = batch
            for width_mult in args.width_mult_list:
                # Step8: Broadcast supernet config from width_mult,
                # and use this config in supernet training.
                net_config = utils.dynabert_config(ofa_model, width_mult)
                ofa_model.set_net_config(net_config)
                logits, teacher_logits = ofa_model(
                    input_ids, segment_ids, attention_mask=[None, None])
                rep_loss = ofa_model.calc_distill_loss()
                if args.task_name == 'sts-b':
                    logit_loss = paddle.zeros(shape=[1], dtype='float32')
                else:
                    logit_loss = soft_cross_entropy(logits,
                                                    teacher_logits.detach())
                loss = rep_loss + args.lambda_logit * logit_loss
                loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % args.logging_steps == 0:
                if paddle.distributed.get_rank() == 0:
                    logger.info(
                        "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                        % (global_step, epoch, step, loss,
                           args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()

            if global_step % args.save_steps == 0:
                tic_eval = time.time()
                evaluate(
                    epoch,
                    global_step,
                    teacher_model,
                    criterion,
                    metric,
                    dev_data_loader,
                    width_mult=100)
                print("eval done total : %s s" % (time.time() - tic_eval))
                for idx, width_mult in enumerate(args.width_mult_list):
                    net_config = utils.dynabert_config(ofa_model, width_mult)
                    ofa_model.set_net_config(net_config)
                    tic_eval = time.time()

                    eval_result, eval_loss = \
                        evaluate(epoch, global_step, ofa_model, criterion, metric, dev_data_loader, width_mult)
                    print("eval done total : %s s" %
                          (time.time() - tic_eval))

                    output_dir = os.path.join(args.output_dir, "model_%s" % width_mult)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if paddle.distributed.get_rank() == 0:
                        if eval_result[2] > best_f1[idx]:
                            best_f1[idx] = eval_result[2]
                            # need better way to get inner model of DataParallel
                            model_to_save = model._layers if isinstance(
                                model, paddle.DataParallel) else model
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)
                            with open(os.path.join(output_dir, "train_result.txt"), 'a', encoding='utf-8') as fp:
                                fp.write(
                                    'epoch:%s, step: %s, width_mult: %s, '
                                    'eval loss: %f,precision: %s, recall: %s, f1: %s,\n'
                                    % (epoch,
                                       global_step,
                                       width_mult,
                                       eval_loss,
                                       eval_result[0],
                                       eval_result[1],
                                       eval_result[2],
                                       ))
            if global_step >= num_training_steps:
                return


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    do_train(args)
