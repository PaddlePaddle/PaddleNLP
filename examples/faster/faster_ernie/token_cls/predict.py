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

import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.experimental import FasterErnieForTokenClassification, to_tensor

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default="ckpt/model_4221", help="The path to model parameters to be loaded.")
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
        for pred in preds:
            # drop the concated CLS and SEP token label
            pred = pred[1:-1]
            label = [label_map[i] for i in pred]
            results.append(label)

    return results


if __name__ == "__main__":
    paddle.set_device(args.device)

    test_ds = load_dataset('msra_ner', splits=('test'))
    texts = ["".join(example["tokens"]) for example in test_ds]
    label_map = dict(enumerate(test_ds.label_list))
    model = FasterErnieForTokenClassification.from_pretrained(
        args.save_dir,
        num_classes=len(test_ds.label_list),
        max_seq_len=args.max_seq_length,
        is_split_into_words=True)
    results = predict(model, texts, label_map, args.batch_size)

    for idx, text in enumerate(texts):
        seq_len = len(text)
        label = results[idx][:seq_len]
        print(text, " : ", " ".join(label))
