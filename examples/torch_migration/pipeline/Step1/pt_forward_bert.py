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
from reprod_log import ReprodLogger
import torch

CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
CONFIG_PATH = CURRENT_DIR.rsplit('/', 1)[0]
sys.path.append(CONFIG_PATH)

from models.pt_bert import BertConfig, BertForSequenceClassification

if __name__ == "__main__":
    # def logger
    reprod_logger = ReprodLogger()

    pytorch_dump_path = '../weights/torch_weight.bin'
    config = BertConfig()
    model = BertForSequenceClassification(config)
    checkpoint = torch.load(pytorch_dump_path)
    model.bert.load_state_dict(checkpoint)

    classifier_weights = torch.load(
        "../classifier_weights/torch_classifier_weights.bin")
    model.load_state_dict(classifier_weights, strict=False)
    model.eval()

    # read or gen fake data
    fake_data = np.load("../fake_data/fake_data.npy")
    fake_data = torch.from_numpy(fake_data)
    # forward
    out = model(fake_data)[0]
    reprod_logger.add("logits", out.cpu().detach().numpy())
    reprod_logger.save("forward_torch.npy")
