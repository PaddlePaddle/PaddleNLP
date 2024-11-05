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

import argparse
import os

import psutil
from predictor import Predictor

from paddlenlp.datasets import load_dataset

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_path_prefix", type=str, required=True, help="The path prefix of inference model to be used.")
parser.add_argument('--model_name_or_path', default="ernie-3.0-medium-zh", help="Select model to train, defaults to ernie-3.0-medium-zh.",
                    choices=["ernie-1.0-large-zh-cw", "ernie-3.0-xbase-zh", "ernie-3.0-base-zh", "ernie-3.0-medium-zh", "ernie-3.0-micro-zh", "ernie-3.0-mini-zh", "ernie-3.0-nano-zh", "ernie-2.0-base-en", "ernie-2.0-large-en", "ernie-m-base", "ernie-m-large"])
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--use_fp16", action='store_true', help="Whether to use fp16 inference, only takes effect when deploying on gpu.")
parser.add_argument("--use_quantize", action='store_true', help="Whether to use quantization for acceleration, only takes effect when deploying on cpu.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for predicting.")
parser.add_argument("--num_threads", default=psutil.cpu_count(logical=False), type=int, help="num_threads for cpu, only takes effect when deploying on cpu.")
parser.add_argument('--device', default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument('--device_id', default=0, help="Select which gpu device to train model.")
parser.add_argument("--perf", action='store_true', help="Whether to compute the latency and f1 score of the test set.")
parser.add_argument("--dataset_dir", required=True, default=None, type=str, help="The dataset directory including data.txt, taxonomy.txt, test.txt(optional, if evaluate the performance).")
parser.add_argument("--perf_dataset", choices=['dev', 'test'], default='dev', type=str, help="evaluate the performance on dev dataset or test dataset")
parser.add_argument('--multilingual', action='store_true', help='Whether is multilingual task')
args = parser.parse_args()
# yapf: enable


def read_local_dataset(path, label_list):
    label_list_dict = {label_list[i]: i for i in range(len(label_list))}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items = line.strip().split("\t")
            if len(items) == 0:
                continue
            elif len(items) == 1:
                sentence = items[0]
                labels = []
            else:
                sentence = "".join(items[:-1])
                label = items[-1]
                labels = [label_list_dict[l] for l in label.split(",")]
            yield {"sentence": sentence, "label": labels}


if __name__ == "__main__":

    label_list = []
    label_dir = os.path.join(args.dataset_dir, "label.txt")
    with open(label_dir, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            label_list.append(line.strip())
    f.close()

    predictor = Predictor(args, label_list)

    if args.perf:
        eval_dir = os.path.join(args.dataset_dir, "{}.txt".format(args.perf_dataset))
        eval_ds = load_dataset(read_local_dataset, path=eval_dir, label_list=label_list, lazy=False)
        texts, labels = predictor.get_text_and_label(eval_ds)

        # preprocess & evaluate & latency
        preprocess_result = predictor.preprocess(texts)
        predictor.evaluate(preprocess_result, labels)
        predictor.performance(preprocess_result)
    else:
        data = []
        data_dir = os.path.join(args.dataset_dir, "data.txt")
        with open(data_dir, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                data.append(line.strip())
        f.close()
        predictor.predict(data)
