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
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.nn.initializer as I
from paddle.metric import Metric, Accuracy, Precision, Recall
from paddlenlp.metrics import AccuracyAndF1
from paddlenlp.embeddings import TokenEmbedding

from args import parse_args
from data import create_data_loader_for_small_model, create_pair_loader_for_small_model

METRIC_CLASSES = {
    "sst-2": Accuracy,
    "qqp": AccuracyAndF1,
    "chnsenticorp": Accuracy
}


class BiLSTM(nn.Layer):

    def __init__(self,
                 embed_dim,
                 hidden_size,
                 vocab_size,
                 output_dim,
                 vocab_path,
                 padding_idx=0,
                 num_layers=1,
                 dropout_prob=0.0,
                 init_scale=0.1,
                 embedding_name=None):
        super(BiLSTM, self).__init__()
        if embedding_name is not None:
            self.embedder = TokenEmbedding(embedding_name,
                                           extended_vocab_path=vocab_path,
                                           keep_extended_vocab_only=True)
            embed_dim = self.embedder.embedding_dim
        else:
            self.embedder = nn.Embedding(vocab_size, embed_dim, padding_idx)

        self.lstm = nn.LSTM(embed_dim,
                            hidden_size,
                            num_layers,
                            'bidirectional',
                            dropout=dropout_prob)

        self.fc = nn.Linear(
            hidden_size * 2,
            hidden_size,
            weight_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)))

        self.fc_1 = nn.Linear(
            hidden_size * 8,
            hidden_size,
            weight_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)))

        self.output_layer = nn.Linear(
            hidden_size,
            output_dim,
            weight_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-init_scale, high=init_scale)))

    def forward(self, x_1, seq_len_1, x_2=None, seq_len_2=None):
        x_embed_1 = self.embedder(x_1)
        lstm_out_1, (hidden_1, _) = self.lstm(x_embed_1,
                                              sequence_length=seq_len_1)
        out_1 = paddle.concat((hidden_1[-2, :, :], hidden_1[-1, :, :]), axis=1)
        if x_2 is not None:
            x_embed_2 = self.embedder(x_2)
            lstm_out_2, (hidden_2, _) = self.lstm(x_embed_2,
                                                  sequence_length=seq_len_2)
            out_2 = paddle.concat((hidden_2[-2, :, :], hidden_2[-1, :, :]),
                                  axis=1)
            out = paddle.concat(
                x=[out_1, out_2, out_1 + out_2,
                   paddle.abs(out_1 - out_2)],
                axis=1)
            out = paddle.tanh(self.fc_1(out))
        else:
            out = paddle.tanh(self.fc(out_1))
        logits = self.output_layer(out)

        return logits


def evaluate(task_name, model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        if task_name == 'qqp':
            input_ids_1, seq_len_1, input_ids_2, seq_len_2, labels = batch
            logits = model(input_ids_1, seq_len_1, input_ids_2, seq_len_2)
        else:
            input_ids, seq_len, labels = batch
            logits = model(input_ids, seq_len)
        loss = loss_fct(logits, labels)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    if isinstance(metric, AccuracyAndF1):
        print(
            "eval loss: %f, acc: %s, precision: %s, recall: %s, f1: %s, acc and f1: %s, "
            % (
                loss.numpy(),
                res[0],
                res[1],
                res[2],
                res[3],
                res[4],
            ),
            end='')
    else:
        print("eval loss: %f, acc: %s, " % (loss.numpy(), res), end='')
    model.train()
    return res[0] if isinstance(metric, AccuracyAndF1) else res


def do_train(args):
    device = paddle.set_device(args.device)
    metric_class = METRIC_CLASSES[args.task_name]
    metric = metric_class()
    if args.task_name == 'qqp':
        train_data_loader, dev_data_loader = create_pair_loader_for_small_model(
            task_name=args.task_name,
            vocab_path=args.vocab_path,
            model_name=args.model_name,
            batch_size=args.batch_size)
    else:
        train_data_loader, dev_data_loader = create_data_loader_for_small_model(
            task_name=args.task_name,
            vocab_path=args.vocab_path,
            model_name=args.model_name if args.task_name == 'sst-2' else None,
            batch_size=args.batch_size)

    model = BiLSTM(args.emb_dim, args.hidden_size, args.vocab_size,
                   args.output_dim, args.vocab_path, args.padding_idx,
                   args.num_layers, args.dropout_prob, args.init_scale,
                   args.embedding_name)

    loss_fct = nn.CrossEntropyLoss()

    if args.optimizer == 'adadelta':
        optimizer = paddle.optimizer.Adadelta(learning_rate=args.lr,
                                              rho=0.95,
                                              parameters=model.parameters())
    else:
        optimizer = paddle.optimizer.Adam(learning_rate=args.lr,
                                          parameters=model.parameters())

    if args.init_from_ckpt:
        model.set_state_dict(paddle.load(args.init_from_ckpt + ".pdparams"))
        optimizer.set_state_dict(paddle.load(args.init_from_ckpt + ".pdopt"))
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.max_epoch):
        for i, batch in enumerate(train_data_loader):
            global_step += 1
            if args.task_name == 'qqp':
                input_ids_1, seq_len_1, input_ids_2, seq_len_2, labels = batch
                logits = model(input_ids_1, seq_len_1, input_ids_2, seq_len_2)
            else:
                input_ids, seq_len, labels = batch
                logits = model(input_ids, seq_len)

            loss = loss_fct(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            if global_step % args.log_freq == 0:
                with paddle.no_grad():
                    print(
                        "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.4f step/s"
                        % (global_step, epoch, i, loss, args.log_freq /
                           (time.time() - tic_train)))
                    tic_eval = time.time()

                    acc = evaluate(args.task_name, model, loss_fct, metric,
                                   dev_data_loader)
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
