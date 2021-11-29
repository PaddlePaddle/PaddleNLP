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

import math, os
from importlib import import_module

import paddle
from paddle import nn
import numpy as np

from model import MemN2N
from data import load_data
from config import Config


@paddle.no_grad()
def eval(model: MemN2N, data, config, mode="Test"):
    """
    evaluate the model performance

    Args:
        model (MemN2N): the model to be evaluate
        data: evaluation data
        config: model and eval configs
        mode: Valid or Test
    
    Returns:
        average loss
    """
    model.eval()
    lossfn = nn.CrossEntropyLoss(reduction='sum')
    N = int(math.ceil(len(data) / config.batch_size))
    total_loss = 0

    context = np.ndarray([config.batch_size, config.mem_size], dtype=np.int64)
    target = np.ndarray([config.batch_size], dtype=np.int64)

    if config.show:
        ProgressBar = getattr(import_module('utils'), 'ProgressBar')
        bar = ProgressBar(mode, max=N - 1)

    m = config.mem_size
    for batch in range(N):
        if config.show:
            bar.next()

        for i in range(config.batch_size):
            if m >= len(data):
                break
            target[i] = data[m]
            context[i, :] = data[m - config.mem_size:m]
            m += 1
        if m >= len(data):
            break

        batch_data = paddle.to_tensor(context)
        batch_label = paddle.to_tensor(target)

        preict = model(batch_data)
        loss = lossfn(preict, batch_label)

        total_loss += loss

    if config.show:
        bar.finish()

    return total_loss / N / config.batch_size


def test(model: MemN2N, test_data, config):
    """
    test the model performance
    """
    test_loss = eval(model, test_data, config, "Test")
    test_perplexity = math.exp(test_loss)
    print("Perplexity on Test: %f" % test_perplexity)


if __name__ == '__main__':
    config = Config("config.yaml")

    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    word2idx, train_data, valid_data, test_data = load_data(config)
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    config.nwords = len(word2idx)

    print("vacab size is %d" % config.nwords)

    model = MemN2N(config)

    model_path = os.path.join(config.checkpoint_dir, config.model_name)
    state_dict = paddle.load(model_path)
    model.set_dict(state_dict)
    test(model, test_data, config)
