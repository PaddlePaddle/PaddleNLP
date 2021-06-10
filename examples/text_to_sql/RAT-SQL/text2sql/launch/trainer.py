#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import os
import traceback
import logging
from pathlib import Path

import paddle
from paddle import nn

from text2sql import utils
from text2sql import io
from text2sql.utils import metrics
from text2sql.launch import infer


def log_train_step(epoch, batch, steps_loss, cost_time):
    if len(steps_loss) == 0:
        return

    logging.info(f'[train] epoch {epoch}, batch {batch}. ' + \
                 f'loss is {sum(steps_loss) / len(steps_loss):.10f}. ' + \
                 f'cost {cost_time:.2f}s')
    steps_loss.clear()


def epoch_train(config, model, optimizer, epoch, train_data, is_debug=False):
    model.train()

    total_loss = 0
    steps_loss = []
    timer = utils.Timer()
    batch_id = 1
    for batch_id, (inputs, labels) in enumerate(train_data(), start=1):
        loss = model(inputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        if type(optimizer._learning_rate) is not float:
            optimizer._learning_rate.step()

        total_loss += loss.numpy().item()
        steps_loss.append(loss.numpy().item())
        if batch_id % config.train.log_steps == 0 or is_debug:
            log_train_step(epoch, batch_id, steps_loss, timer.interval())
    log_train_step(epoch, batch_id, steps_loss, timer.interval())

    return total_loss / batch_id


def _eval_during_train(model, data, epoch, output_root):
    if epoch in [1, 2, 3, 4] + \
                [6, 7, 9, 10, 11, 13, 14, 16, 17, 19] + \
                list(range(21, 100, 2)):
        return 0, epoch
    model.eval()
    try:
        output = Path(
            output_root
        ) / 'infer_result' / f'{data.name}.infer_epoch{epoch:03d}.sql'
        infer.inference(model, data, output)
    except OSError as ose:
        traceback.print_exc()
        logging.error(traceback.format_exc())
        return 0, epoch

    mean_loss = 0
    return mean_loss, epoch


def train(config,
          model,
          optimizer,
          epochs,
          train_data,
          dev_data,
          test_data=None):
    best_acc = -1e10
    best_epoch = 0
    timer = utils.Timer()
    for epoch in range(1, epochs + 1):
        loss = epoch_train(config, model, optimizer, epoch, train_data,
                           config.general.is_debug)
        cost_time = timer.interval()
        logging.info(
            f'[train] epoch {epoch}/{epochs} loss is {loss:.6f}, cost {cost_time:.2f}s.'
        )

        dev_loss, dev_acc = _eval_during_train(model, dev_data, epoch,
                                               config.data.output)
        log_str = f'[eval] dev loss {dev_loss:.6f}, acc {dev_acc:.4f}.'
        if test_data is not None:
            test_loss, test_acc = _eval_during_train(model, test_data, epoch,
                                                     config.data.output)
            log_str += f' test loss {test_loss:.6f}, acc {test_acc:.4f}.'

        if dev_acc > best_acc:
            best_acc, best_epoch = dev_acc, epoch
            save_path = os.path.join(config.data.output,
                                     f'epoch{epoch:03d}_acc{best_acc:.4f}',
                                     'model')
            io.save(model, optimizer, save_path)
            log_str += ' got best and saved.'
        else:
            log_str += f' best acc is {best_acc} on epoch {best_epoch}.'

        cost_time = timer.interval()
        log_str += f' cost [{cost_time:.2f}s]'
        logging.info(log_str)


if __name__ == "__main__":
    """run some simple test cases"""
    pass
