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

import numpy as np

import paddle
import paddle.nn.functional as F
from paddlenlp.data import Tuple, Pad
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--params_path",
                    default="./checkpoint/",
                    type=str,
                    help="The path to model parameters to be loaded.")
parser.add_argument("--dataset_dir",
                    default=None,
                    type=str,
                    help="Local dataset directory should"
                    "include data.txt and label.txt")
parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="The maximum total input sequence length "
                    "after tokenization. Sequences longer than this"
                    "will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument('--device',
                    choices=['cpu', 'gpu', 'xpu', 'npu'],
                    default="gpu",
                    help="Select which device to train model, defaults to gpu.")

args = parser.parse_args()


@paddle.no_grad()
def predict(data, label_list):
    """
    Predicts the data labels.
    Args:

        data (obj:`List`): The processed data whose each element is one sequence.
        label_map(obj:`List`): The label id (key) to label str (value) map.
 
    """
    paddle.set_device(args.device)
    model = AutoModelForSequenceClassification.from_pretrained(args.params_path)
    tokenizer = AutoTokenizer.from_pretrained(args.params_path)

    examples = []
    for text in data:
        result = tokenizer(text=text, max_seq_len=args.max_seq_length)
        examples.append((result['input_ids'], result['token_type_ids']))

    # Seperates data into some batches.
    batches = [
        examples[i:i + args.batch_size]
        for i in range(0, len(examples), args.batch_size)
    ]

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    ): fn(samples)

    results = []
    model.eval()
    for batch in batches:
        input_ids, token_type_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_list[i] for i in idx]
        results.extend(labels)

    for text, r in zip(data, results):
        print("input data:", text)
        print('label: {}'.format(r))
        print('---------------------------------')
    return


if __name__ == "__main__":
    if args.dataset_dir is not None:
        data_dir = os.path.join(args.dataset_dir, "data.txt")
        label_dir = os.path.join(args.dataset_dir, "label.txt")
        label_list = []
        data = []
        with open(label_dir, 'r', encoding='utf-8') as f:
            for line in f:
                label_list.append(line.strip())
        f.close()
        with open(data_dir, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(line.strip())
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
