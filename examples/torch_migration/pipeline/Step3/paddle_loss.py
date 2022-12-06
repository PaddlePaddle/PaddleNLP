# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
import paddle.nn as nn
from reprod_log import ReprodLogger

CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
CONFIG_PATH = CURRENT_DIR.rsplit("/", 1)[0]
sys.path.append(CONFIG_PATH)

from models.pd_bert import BertConfig, BertForSequenceClassification

if __name__ == "__main__":
    paddle.set_device("cpu")

    # def logger
    reprod_logger = ReprodLogger()

    paddle_dump_path = "../weights/paddle_weight.pdparams"
    config = BertConfig()
    model = BertForSequenceClassification(config)
    checkpoint = paddle.load(paddle_dump_path)
    model.bert.load_dict(checkpoint)

    classifier_weights = paddle.load("../classifier_weights/paddle_classifier_weights.bin")
    model.load_dict(classifier_weights)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # read or gen fake data
    fake_data = np.load("../fake_data/fake_data.npy")
    fake_data = paddle.to_tensor(fake_data)

    fake_label = np.load("../fake_data/fake_label.npy")
    fake_label = paddle.to_tensor(fake_label)

    # forward
    out = model(fake_data)[0]

    loss = criterion(out, fake_label)
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("loss_paddle.npy")
