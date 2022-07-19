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

import os
import argparse

import psutil
import paddle
from paddlenlp.utils.log import logger
from paddlenlp.datasets import load_dataset

from predictor import Predictor

parser = argparse.ArgumentParser()
parser.add_argument("--model_path_prefix",
                    type=str,
                    required=True,
                    help="The path prefix of inference model to be used.")
parser.add_argument("--model_name_or_path",
                    default="ernie-3.0-base-zh",
                    type=str,
                    help="The directory or name of model.")
parser.add_argument("--dataset",
                    default="cblue",
                    type=str,
                    help="Dataset for text classfication.")
parser.add_argument("--task_name",
                    default="KUAKE-QIC",
                    type=str,
                    help="Task name for text classfication dataset.")
parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="The maximum total input sequence length after "
                    "tokenization. Sequences longer than this will "
                    "be truncated, sequences shorter will be padded.")
parser.add_argument("--use_fp16",
                    action='store_true',
                    help="Whether to use fp16 inference, only "
                    "takes effect when deploying on gpu.")
parser.add_argument("--use_quantize",
                    action='store_true',
                    help="Whether to use quantization for acceleration,"
                    " only takes effect when deploying on cpu.")
parser.add_argument("--batch_size",
                    default=200,
                    type=int,
                    help="Batch size per GPU/CPU for predicting.")
parser.add_argument("--num_threads",
                    default=psutil.cpu_count(logical=False),
                    type=int,
                    help="num_threads for cpu, only takes effect"
                    " when deploying on cpu.")
parser.add_argument('--device',
                    choices=['cpu', 'gpu'],
                    default="gpu",
                    help="Select which device to train model, defaults to gpu.")
parser.add_argument('--device_id',
                    default=0,
                    help="Select which gpu device to train model.")
parser.add_argument("--perf",
                    action='store_true',
                    help="Whether to compute the latency "
                    "and f1 score of the test set.")
parser.add_argument("--dataset_dir",
                    default=None,
                    type=str,
                    help="The dataset directory including "
                    "data.txt, taxonomy.txt, test.txt/dev.txt(optional),"
                    "if evaluate the performance).")
parser.add_argument("--perf_dataset",
                    choices=['dev', 'test'],
                    default='test',
                    type=str,
                    help="evaluate the performance on"
                    "dev dataset or test dataset")
args = parser.parse_args()


def read_local_dataset(path, label_list):
    label_list_dict = {label_list[i]: i for i in range(len(label_list))}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sentence, label = line.strip().split('\t')
            yield {'text_a': sentence, 'label': label_list_dict[label]}


def predict(data, label_list):
    """
    Predicts the data labels.
    Args:

        data (obj:`List`): The processed data whose each element is one sequence.
        label_map(obj:`List`): The label id (key) to label str (value) map.
 
    """
    predictor = Predictor(args, label_list)
    predictor.predict(data)

    if args.perf:

        if args.dataset_dir is not None:
            eval_dir = os.path.join(args.dataset_dir,
                                    "{}.txt".format(args.perf_dataset))
            eval_ds = load_dataset(read_local_dataset,
                                   path=eval_dir,
                                   label_list=label_list,
                                   lazy=False)
        else:
            eval_ds = load_dataset(args.dataset,
                                   name=args.task_name,
                                   splits=[args.perf_dataset])

        texts, labels = predictor.get_text_and_label(eval_ds)

        preprocess_result = predictor.preprocess(texts)

        # evaluate
        predictor.evaluate(preprocess_result, labels)

        # latency
        predictor.performance(preprocess_result)


if __name__ == "__main__":

    if args.dataset_dir is not None:
        data_dir = os.path.join(args.dataset_dir, "data.txt")
        label_dir = os.path.join(args.dataset_dir, "label.txt")

        data = []
        label_list = []

        with open(data_dir, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                data.append(line.strip())
        f.close()

        with open(label_dir, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                label_list.append(line.strip())
        f.close()
    else:
        data = [
            "黑苦荞茶的功效与作用及食用方法", "交界痣会凸起吗", "检查是否能怀孕挂什么科", "鱼油怎么吃咬破吃还是直接咽下去",
            "幼儿挑食的生理原因是"
        ]
        label_list = [
            '病情诊断', '治疗方案', '病因分析', '指标解读', '就医建议', '疾病表述', '后果表述', '注意事项',
            '功效作用', '医疗费用', '其他'
        ]
    predict(data, label_list)
