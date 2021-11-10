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

import paddle
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
from paddlenlp.experimental import FasterErnieModel, FasterErnieForSequenceClassification, to_tensor

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default="ckpt/model_900", help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", type=int, default=128, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def predict(model, data, label_map, batch_size=1):
    # Seperates data into some batches.
    batches = [
        data[idx:idx + batch_size] for idx in range(0, len(data), batch_size)
    ]

    results = []
    model.eval()
    for texts in batches:
        texts = to_tensor(texts)
        logits, preds = model(texts)
        preds = preds.numpy()
        labels = [label_map[i] for i in preds]
        results.extend(labels)
    return results


if __name__ == "__main__":
    paddle.set_device(args.device)
    test_ds = load_dataset("chnsenticorp", splits=["test"])
    data = [example["text"] for example in test_ds]
    label_map = {0: 'negative', 1: 'positive'}

    model = FasterErnieForSequenceClassification.from_pretrained(
        args.save_dir,
        num_classes=len(test_ds.label_list),
        max_seq_len=args.max_seq_length)
    results = predict(model, data, label_map, batch_size=args.batch_size)

    for idx, text in enumerate(data):
        print(text, " : ", results[idx])
