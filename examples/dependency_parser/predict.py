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
import os
import time

import numpy as np
import paddle
import paddlenlp as ppnlp

from env import Environment
from data import batchify, TextDataset
from model.model import DDParserModel

# yapf: disable
parser = argparse.ArgumentParser()
#parser.add_argument("--params_path", type=str, required=True, help="The path to model parameters to be loaded.")
parser.add_argument("--device", choices=["cpu", "gpu", "xpu"], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for predicting.")
parser.add_argument("--data_path", type=str, default="./data", help="The path of datasets to be loaded")
parser.add_argument("--save_dir", type=str, default='checkpoints/', help="Directory of saved model checkpoint.")
parser.add_argument("--mode", choices=["predict_file", "predict_text"], default="predict_file", help="Select predict mode, defaults to predict the file.")

args = parser.parse_args()
# yapf: enable

@paddle.no_grad()
def predict():
    pass
    #return results

def do_predict_file(env):
    args = env.args

    file_name = "test.txt"
    predict = Corpus.load(os.path.join(args.data_path, file_name), env.fields)
    predict_ds = TextDataset(predict, [env.WORD, env.FEAT], args.n_buckets)

    #return results

def do_predict_text(env):
    pass
    

if __name__ == "__main__":
    env = Environment(args)
    do_predict_file(env)
