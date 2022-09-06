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

import paddle
import paddle.nn as nn
from paddle.metric import Accuracy

from paddlenlp.transformers import BertForSequenceClassification
from paddlenlp.metrics import AccuracyAndF1

from args import parse_args
from small import BiLSTM
from data import create_distill_loader

METRIC_CLASSES = {
    "sst-2": Accuracy,
    "qqp": AccuracyAndF1,
    "chnsenticorp": Accuracy
}


class TeacherModel(object):

    def __init__(self, teacher_dir):
        self.model = BertForSequenceClassification.from_pretrained(teacher_dir)
        self.model.eval()


def evaluate(task_name, model, metric, data_loader):
    model.eval()
    metric.reset()
    for i, batch in enumerate(data_loader):
        if task_name == 'qqp':
            _, _, student_input_ids_1, seq_len_1, student_input_ids_2, seq_len_2, labels = batch
            logits = model(student_input_ids_1, seq_len_1, student_input_ids_2,
                           seq_len_2)
        else:
            _, _, student_input_ids, seq_len, labels = batch
            logits = model(student_input_ids, seq_len)

        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    if isinstance(metric, AccuracyAndF1):
        print("acc: %s, precision: %s, recall: %s, f1: %s, acc and f1: %s, " % (
            res[0],
            res[1],
            res[2],
            res[3],
            res[4],
        ),
              end='')
    else:
        print("acc: %s, " % (res), end='')
    model.train()


def do_train(agrs):
    device = paddle.set_device(args.device)
    train_data_loader, dev_data_loader = create_distill_loader(
        args.task_name,
        model_name=args.model_name,
        vocab_path=args.vocab_path,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        n_iter=args.n_iter,
        whole_word_mask=args.whole_word_mask,
        seed=args.seed)

    model = BiLSTM(args.emb_dim, args.hidden_size, args.vocab_size,
                   args.output_dim, args.vocab_path, args.padding_idx,
                   args.num_layers, args.dropout_prob, args.init_scale,
                   args.embedding_name)

    if args.optimizer == 'adadelta':
        optimizer = paddle.optimizer.Adadelta(learning_rate=args.lr,
                                              rho=0.95,
                                              parameters=model.parameters())
    else:
        optimizer = paddle.optimizer.Adam(learning_rate=args.lr,
                                          parameters=model.parameters())

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    metric_class = METRIC_CLASSES[args.task_name]
    metric = metric_class()

    teacher = TeacherModel(args.teacher_dir)

    print("Start to distill student model.")

    if args.init_from_ckpt:
        model.set_state_dict(paddle.load(args.init_from_ckpt + ".pdparams"))
        optimizer.set_state_dict(paddle.load(args.init_from_ckpt + ".pdopt"))
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.max_epoch):
        model.train()
        for i, batch in enumerate(train_data_loader):
            global_step += 1
            if args.task_name == 'qqp':
                bert_input_ids, bert_segment_ids, student_input_ids_1, seq_len_1, student_input_ids_2, seq_len_2, labels = batch
            else:
                bert_input_ids, bert_segment_ids, student_input_ids, seq_len, labels = batch

            # Calculate teacher model's forward.
            with paddle.no_grad():
                teacher_logits = teacher.model(bert_input_ids, bert_segment_ids)

            # Calculate student model's forward.
            if args.task_name == 'qqp':
                logits = model(student_input_ids_1, seq_len_1,
                               student_input_ids_2, seq_len_2)
            else:
                logits = model(student_input_ids, seq_len)

            loss = args.alpha * ce_loss(logits, labels) + (
                1 - args.alpha) * mse_loss(logits, teacher_logits)

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            if global_step % args.log_freq == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.4f step/s"
                    % (global_step, epoch, i, loss, args.log_freq /
                       (time.time() - tic_train)))
                tic_eval = time.time()
                acc = evaluate(args.task_name, model, metric, dev_data_loader)
                print("eval done total : %s s" % (time.time() - tic_eval))
                tic_train = time.time()

            if global_step % args.save_steps == 0:
                paddle.save(
                    model.state_dict(),
                    os.path.join(args.output_dir,
                                 "step_" + str(global_step) + ".pdparams"))
                paddle.save(
                    optimizer.state_dict(),
                    os.path.join(args.output_dir,
                                 "step_" + str(global_step) + ".pdopt"))


if __name__ == '__main__':
    args = parse_args()
    print(args)
    paddle.seed(args.seed)
    do_train(args)
