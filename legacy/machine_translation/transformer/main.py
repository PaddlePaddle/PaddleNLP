# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import logging

import numpy as np
import paddle
import paddle.fluid as fluid

from utils.configure import PDConfig
from train import do_train
from predict import do_predict
from inference_model import do_save_inference_model

if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    LOG_FORMAT = "[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(
        stream=sys.stdout, level=logging.DEBUG, format=LOG_FORMAT)
    logging.getLogger().setLevel(logging.INFO)

    args = PDConfig(yaml_file="./transformer.yaml")
    args.build()
    args.Print()

    if args.do_train:
        do_train(args)

    if args.do_predict:
        do_predict(args)

    if args.do_save_inference_model:
        do_save_inference_model(args)
