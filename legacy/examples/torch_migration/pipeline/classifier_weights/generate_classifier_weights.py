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


def generate(seed):
    np.random.seed(seed)
    weight = np.random.normal(0, 0.02, (768, 2)).astype("float32")
    bias = np.zeros((2,)).astype("float32")
    paddle_weights = {
        "classifier.weight": weight,
        "classifier.bias": bias,
    }
    torch_weights = {
        "classifier.weight": torch.from_numpy(weight).t(),
        "classifier.bias": torch.from_numpy(bias),
    }
    torch.save(torch_weights, "torch_classifier_weights.bin")
    paddle.save(paddle_weights, "paddle_classifier_weights.bin")


if __name__ == "__main__":
    generate(seed=42)
