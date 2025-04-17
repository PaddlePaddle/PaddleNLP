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

import numpy as np
import paddle
import torch
from datasets import load_metric
from paddle.metric import Accuracy
from reprod_log import ReprodLogger


def generate():
    pd_metric = Accuracy()
    pd_metric.reset()
    hf_metric = load_metric("accuracy.py")
    for i in range(4):
        logits = np.random.normal(0, 1, size=(64, 2)).astype("float32")
        labels = np.random.randint(0, 2, size=(64,)).astype("int64")
        # paddle metric
        correct = pd_metric.compute(paddle.to_tensor(logits), paddle.to_tensor(labels))
        pd_metric.update(correct)
        # hf metric
        hf_metric.add_batch(
            predictions=torch.from_numpy(logits).argmax(dim=-1),
            references=torch.from_numpy(labels),
        )
    pd_accuracy = pd_metric.accumulate()
    hf_accuracy = hf_metric.compute()["accuracy"]
    reprod_logger = ReprodLogger()
    reprod_logger.add("accuracy", np.array([pd_accuracy]))
    reprod_logger.save("metric_paddle.npy")
    reprod_logger = ReprodLogger()
    reprod_logger.add("accuracy", np.array([hf_accuracy]))
    reprod_logger.save("metric_torch.npy")


if __name__ == "__main__":
    generate()
