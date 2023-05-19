# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from functools import partial

import paddle
from data import convert_example_to_feature, load_dict
from datasets import load_dataset
from tqdm import tqdm

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.metrics.glue import AccuracyAndF1
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer


def evaluate(model, data_loader, metric):

    model.eval()
    metric.reset()
    for batch_data in tqdm(data_loader):
        input_ids, token_type_ids, labels = batch_data["input_ids"], batch_data["token_type_ids"], batch_data["labels"]
        logits = model(input_ids, token_type_ids=token_type_ids)
        correct = metric.compute(logits, labels)
        metric.update(correct)

    accuracy, precision, recall, f1, _ = metric.accumulate()

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="The path of saved model that you want to load.")
    parser.add_argument('--test_path', type=str, default=None, help="The path of test set.")
    parser.add_argument("--label_path", type=str, default=None, help="The path of label dict.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type=int, default=512, help="The maximum total input sequence length after tokenization.")
    args = parser.parse_args()
    # yapf: enbale

    # load dev data
    model_name = "skep_ernie_1.0_large_ch"
    label2id, id2label = load_dict(args.label_path)
    datasets = load_dataset("text", data_files={"test": args.test_path})

    tokenizer = SkepTokenizer.from_pretrained(model_name)
    trans_func = partial(convert_example_to_feature, tokenizer=tokenizer, label2id=label2id, max_seq_len=args.max_seq_len)
    test_ds = datasets["test"].map(trans_func, batched=False, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer)

    test_batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = paddle.io.DataLoader(test_ds, batch_sampler=test_batch_sampler, collate_fn=data_collator)

    # load model
    loaded_state_dict = paddle.load(args.model_path)
    model = SkepForSequenceClassification.from_pretrained(model_name, num_classes=len(label2id))
    model.load_dict(loaded_state_dict)

    metric = AccuracyAndF1()

    # evaluate on dev data
    accuracy, precision, recall, f1 = evaluate(model, test_loader, metric)
    print(f'evaluation result: accuracy:{accuracy:.5f} precision: {precision:.5f}, recall: {recall:.5f},  F1: {f1:.5f}')
