# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
import time
import logging
import random

import numpy as np
import paddle
import paddle.nn as nn
from paddle.metric import Metric, Accuracy, Precision, Recall

from paddlenlp.transformers import BertModel, BertForSequenceClassification
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman

from args import parse_args
from data import create_glue_data_loader, MODEL_CLASSES, METRIC_CLASSES
from kd_loss import cal_pkd_loss

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


def transformer_encoder_layer_forward(self, src, src_mask=None):
    output = src
    cls_token_hidden_states = []
    for i, mod in enumerate(self.layers):
        output = mod(output, src_mask=src_mask)
        cls_token_hidden_states.append(output[:, 0])

    if self.norm is not None:
        output = self.norm(output)

    return output, cls_token_hidden_states


nn.TransformerEncoder.forward = transformer_encoder_layer_forward


def bert_forward(self,
                 input_ids,
                 token_type_ids=None,
                 position_ids=None,
                 attention_mask=None):
    if attention_mask is None:
        attention_mask = paddle.unsqueeze(
            (input_ids == self.pad_token_id
             ).astype(self.pooler.dense.weight.dtype) * -1e9,
            axis=[1, 2])
    embedding_output = self.embeddings(
        input_ids=input_ids,
        position_ids=position_ids,
        token_type_ids=token_type_ids)
    encoder_outputs, cls_token_hidden_states = self.encoder(embedding_output,
                                                            attention_mask)
    sequence_output = encoder_outputs
    pooled_output = self.pooler(sequence_output)
    return sequence_output, pooled_output, cls_token_hidden_states


BertModel.forward = bert_forward


def bert_for_seq_class_forward(self,
                               input_ids,
                               token_type_ids=None,
                               position_ids=None,
                               attention_mask=None):
    _, pooled_output, cls_token_hidden_states = self.bert(
        input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask)

    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    return logits, cls_token_hidden_states


BertForSequenceClassification.forward = bert_for_seq_class_forward


@paddle.no_grad()
def evaluate(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, segment_ids, labels = batch
        logits, _ = model(input_ids, segment_ids)
        loss = loss_fct(logits, labels)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    if isinstance(metric, AccuracyAndF1):
        logger.info(
            "eval loss: %f, acc: %s, precision: %s, recall: %s, f1: %s, acc and f1: %s."
            % (loss.numpy(), res[0], res[1], res[2], res[3], res[4]))
    elif isinstance(metric, Mcc):
        logger.info("eval loss: %f, mcc: %s." % (loss.numpy(), res[0]))
    elif isinstance(metric, PearsonAndSpearman):
        logger.info(
            "eval loss: %f, pearson: %s, spearman: %s, pearson and spearman: %s."
            % (loss.numpy(), res[0], res[1], res[2]))
    else:
        logger.info("eval loss: %f, acc: %s." % (loss.numpy(), res))
    model.train()
    if isinstance(metric, AccuracyAndF1) or isinstance(
            metric, Mcc) or isinstance(metric, PearsonAndSpearman):
        return res[0]
    return res


def do_train(args):
    set_seed(args)

    teacher_model = BertForSequenceClassification.from_pretrained(
        args.teacher_finetuned_model_path)

    student_model = BertForSequenceClassification(
        BertModel(
            vocab_size=30522,
            num_hidden_layers=args.num_layers_of_student_model))
    student_model.set_state_dict(
        paddle.load(
            os.path.join(args.pretrained_model_path, args.model_name_or_path,
                         args.model_name_or_path + ".pdparams")))

    if args.task_name == "mnli":
        train_data_loader, dev_data_loader_matched, dev_data_loader_mismatched = create_glue_data_loader(
            args.task_name)
    else:
        train_data_loader, dev_data_loader = create_glue_data_loader(
            args.task_name)

    num_training_steps = args.max_steps if args.max_steps > 0 else (
        len(train_data_loader) * args.num_train_epochs)
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in student_model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=student_model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    args.task_name = args.task_name.lower()
    metric_class = METRIC_CLASSES[args.task_name]
    args.model_type = args.model_type.lower()
    _, tokenizer_class = MODEL_CLASSES[args.model_type]

    evaluate_loss_fct = nn.CrossEntropyLoss()
    metric = metric_class()
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, segment_ids, labels = batch
            student_logits, student_cls_token_hidden_states = student_model(
                input_ids, segment_ids)
            if args.beta > 1e-7:
                with paddle.no_grad():
                    teacher_logits, teacher_cls_token_hidden_states = teacher_model(
                        input_ids, segment_ids)
                loss, ce_loss, ds_loss, pt_loss = cal_pkd_loss(
                    args, student_logits, labels, teacher_logits,
                    teacher_cls_token_hidden_states,
                    student_cls_token_hidden_states)
                writer.add_scalar(
                    tag="pt_loss", step=global_step, value=pt_loss)

            elif args.alpha > 1e-7:
                with paddle.no_grad():
                    teacher_logits, _ = teacher_model(input_ids, segment_ids)
                loss = cal_pkd_loss(args, student_logits, labels,
                                    teacher_logits)
            else:
                loss = cal_pkd_loss(args, student_logits, labels)
            writer.add_scalar(tag="train_loss", step=global_step, value=loss)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % args.logging_steps == 0:
                logger.info(
                    "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                    % (global_step, num_training_steps, epoch, step,
                       paddle.distributed.get_rank(), loss, optimizer.get_lr(),
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            if global_step % args.save_steps == 0:
                tic_eval = time.time()
                if args.task_name == "mnli":
                    acc = evaluate(student_model, evaluate_loss_fct, metric,
                                   dev_data_loader_matched)
                    evaluate(student_model, evaluate_loss_fct, metric,
                             dev_data_loader_mismatched)
                    logger.info("eval done total : %s s" %
                                (time.time() - tic_eval))
                else:
                    acc = evaluate(student_model, evaluate_loss_fct, metric,
                                   dev_data_loader)
                    logger.info("eval done total : %s s" %
                                (time.time() - tic_eval))
                writer.add_scalar(tag="eval_acc", step=global_step, value=acc)

                if (not args.n_gpu > 1) or paddle.distributed.get_rank() == 0:
                    output_dir = os.path.join(args.output_dir,
                                              "%s_model_%d.pdparams" %
                                              (args.task_name, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Need better way to get inner model of DataParallel
                    model_to_save = student_model._layers if isinstance(
                        student_model, paddle.DataParallel) else student_model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)

    if args.n_gpu > 1:
        paddle.distributed.spawn(do_train, args=(args, ), nprocs=args.n_gpu)
    else:
        do_train(args)
