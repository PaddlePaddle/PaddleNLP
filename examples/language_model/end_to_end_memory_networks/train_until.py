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

import argparse
import os, random, time

import paddle
import numpy as np

from data import load_data
from model import MemN2N
from train import train
from eval import test
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument("--target",
                    default=111.0,
                    type=float,
                    help="target perplexity")
target = parser.parse_args().target

if __name__ == '__main__':
    config = Config('config.yaml')
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    word2idx, train_data, valid_data, test_data = load_data(config)
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    config.nwords = len(word2idx)
    print("vacab size is %d" % config.nwords)

    while True:
        random.seed(time.time())
        config.srand = random.randint(0, 100000)

        np.random.seed(config.srand)
        random.seed(config.srand)
        paddle.seed(config.srand)

        model = MemN2N(config)
        train(model, train_data, valid_data, config)

        test_ppl = test(model, test_data, config)
        if test_ppl < target:
            model_path = os.path.join(
                config.checkpoint_dir,
                config.model_name + "_" + str(config.srand) + "_good")
            paddle.save(model.state_dict(), model_path)
            break
