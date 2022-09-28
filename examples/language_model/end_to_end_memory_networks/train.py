# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from importlib import import_module
import os, math
import random

import paddle
from paddle import nn
import numpy as np

from model import MemN2N
from config import Config
from eval import eval
from data import load_data


def train_single_epoch(model: MemN2N, lr, data, config):
    """
    train one epoch

    Args:
        model (MemN2N): model to be trained
        lr (float): the learning rate of this epoch
        data: training data
        config: configs

    Returns:
        float: average loss
    """
    model.train()
    N = int(math.ceil(len(data) / config.batch_size))  # total train N batchs

    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=config.max_grad_norm)
    optimizer = paddle.optimizer.SGD(learning_rate=lr,
                                     parameters=model.parameters(),
                                     grad_clip=clip)
    lossfn = nn.CrossEntropyLoss(reduction='sum')

    total_loss = 0

    if config.show:
        ProgressBar = getattr(import_module('utils'), 'ProgressBar')
        bar = ProgressBar('Train', max=N)

    for batch in range(N):
        if config.show:
            bar.next()

        optimizer.clear_grad()
        context = np.ndarray([config.batch_size, config.mem_size],
                             dtype=np.int64)
        target = np.ndarray([config.batch_size], dtype=np.int64)
        for i in range(config.batch_size):
            m = random.randrange(config.mem_size, len(data))
            target[i] = data[m]
            context[i, :] = data[m - config.mem_size:m]

        batch_data = paddle.to_tensor(context)
        batch_label = paddle.to_tensor(target)

        preict = model(batch_data)
        loss = lossfn(preict, batch_label)
        loss.backward()
        optimizer.step()
        total_loss += loss

    if config.show:
        bar.finish()

    return total_loss / N / config.batch_size


def train(model: MemN2N, train_data, valid_data, config):
    """
    do train

    Args:
        model (MemN2N): the model to be evaluate
        train_data: training data
        valid_data: validating data
        config: model and training configs
    
    Returns:
        no return
    """
    lr = config.init_lr

    train_losses = []
    train_perplexities = []

    valid_losses = []
    valid_perplexities = []

    for epoch in range(1, config.nepoch + 1):
        train_loss = train_single_epoch(model, lr, train_data, config)
        valid_loss = eval(model, valid_data, config, "Validation")

        info = {'epoch': epoch, 'learning_rate': lr}

        # When the loss on the valid no longer drops, it's like learning rate divided by 1.5
        if len(valid_losses) > 0 and valid_loss > valid_losses[-1] * 0.9999:
            lr /= 1.5

        train_losses.append(train_loss)
        train_perplexities.append(math.exp(train_loss))

        valid_losses.append(valid_loss)
        valid_perplexities.append(math.exp(valid_loss))

        info["train_perplexity"] = train_perplexities[-1]
        info["validate_perplexity"] = valid_perplexities[-1]

        print(info)

        if epoch % config.log_epoch == 0:
            save_dir = os.path.join(config.checkpoint_dir, "model_%d" % epoch)
            paddle.save(model.state_dict(), save_dir)
            lr_path = os.path.join(config.checkpoint_dir, "lr_%d" % epoch)
            with open(lr_path, "w") as f:
                f.write(f"{lr}")

        # to get the target ppl
        if info["validate_perplexity"] < config.target_ppl:
            save_dir = os.path.join(config.checkpoint_dir, "model_good")
            paddle.save(model.state_dict(), save_dir)
            break

        if lr < 1e-5:
            break

    save_dir = os.path.join(config.checkpoint_dir, "model")
    paddle.save(model.state_dict(), save_dir)


if __name__ == '__main__':
    config = Config('config.yaml')

    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    word2idx, train_data, valid_data, test_data = load_data(config)
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    config.nwords = len(word2idx)
    print("vacab size is %d" % config.nwords)

    np.random.seed(config.srand)
    random.seed(config.srand)
    paddle.seed(config.srand)

    model = MemN2N(config)
    if config.recover_train:
        model_path = os.path.join(config.checkpoint_dir, config.model_name)
        state_dict = paddle.load(model_path)
        model.set_dict(state_dict)
    train(model, train_data, valid_data, config)
