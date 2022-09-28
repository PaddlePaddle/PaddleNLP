#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import argparse

import paddle
import paddle.nn as nn
import numpy as np
import pandas as pd

from data import CovidDataset
from model import TCNNetwork

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--data_path", type=str, default="time_series_covid19_confirmed_global.csv", help="The data path.")
parser.add_argument("--seq_length", type=int, default=8, help="The time series length.")
parser.add_argument("--test_data_size", type=int, default=30, help="The number of data used to test.")
parser.add_argument("--use_gpu", action='store_true', default=False, help="If set, use GPU for training.")
parser.add_argument("--init_checkpoint", type=str, default="save_dir/final", help="Path to init model.")
args = parser.parse_args()
# yapf: enable


def test():
    if args.use_gpu:
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")

    test_dataset = CovidDataset(args.data_path,
                                args.test_data_size,
                                args.seq_length,
                                mode="test")

    network = TCNNetwork(input_size=1)

    model = paddle.Model(network)

    model.prepare()

    model.load(args.init_checkpoint)

    preds = model.predict(test_dataset)

    file_path = "results.txt"
    with open(file_path, "w", encoding="utf8") as fout:
        for pred in test_dataset.postprocessing(preds):
            fout.write("%s\n" % str(pred))

    print("The prediction has been saved in %s" % file_path)


if __name__ == "__main__":
    print(args)
    test()
