# C#opyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import sys
from functools import partial
import distutils.util
import numpy as np

import paddle
from paddle.metric import Accuracy
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad

from paddlenlp.transformers import AutoTokenizer
from infer_backend import InferBackend

METRIC_CLASSES = {
    "afqmc": Accuracy,
    "tnews": Accuracy,
    "iflytek": Accuracy,
    "ocnli": Accuracy,
    "cmnli": Accuracy,
    "cluewsc2020": Accuracy,
    "csl": Accuracy,
}


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default='tnews',
        type=str,
        help="The name of the task to perform predict, selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default="ernie-3.0-medium-zh",
        type=str,
        help="The directory or name of model.", )
    parser.add_argument(
        "--model_path",
        default='tnews_quant_models/mse4/int8',
        type=str,
        required=True,
        help="The path prefix of inference model to be used.", )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu", "xpu"],
        help="Device selected for inference.", )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for predict.", )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--perf_warmup_steps",
        default=20,
        type=int,
        help="Warmup steps for performance test.", )
    parser.add_argument(
        "--perf",
        action='store_true',
        help="Whether to test performance.", )
    parser.add_argument(
        "--collect_shape",
        action='store_true',
        help="Whether to collect shape info.", )
    parser.add_argument(
        "--int8",
        action='store_true',
        help="Whether to use int8 inference.", )
    parser.add_argument(
        "--num_threads",
        default=10,
        type=int,
        help="num_threads for cpu.", )
    args = parser.parse_args()
    return args


def convert_example(example,
                    tokenizer,
                    label_list,
                    is_test=False,
                    max_seq_length=512):
    """convert a glue example into necessary features"""
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = np.array(example["label"], dtype="int64")
    # Convert raw text to feature
    if 'keyword' in example:  # CSL
        sentence1 = " ".join(example['keyword'])
        example = {
            'sentence1': sentence1,
            'sentence2': example['abst'],
            'label': example['label']
        }
    elif 'target' in example:  # wsc
        text, query, pronoun, query_idx, pronoun_idx = example['text'], example[
            'target']['span1_text'], example['target']['span2_text'], example[
                'target']['span1_index'], example['target']['span2_index']
        text_list = list(text)
        assert text[pronoun_idx:(pronoun_idx + len(pronoun)
                                 )] == pronoun, "pronoun: {}".format(pronoun)
        assert text[query_idx:(query_idx + len(query)
                               )] == query, "query: {}".format(query)
        if pronoun_idx > query_idx:
            text_list.insert(query_idx, "_")
            text_list.insert(query_idx + len(query) + 1, "_")
            text_list.insert(pronoun_idx + 2, "[")
            text_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
        else:
            text_list.insert(pronoun_idx, "[")
            text_list.insert(pronoun_idx + len(pronoun) + 1, "]")
            text_list.insert(query_idx + 2, "_")
            text_list.insert(query_idx + len(query) + 2 + 1, "_")
        text = "".join(text_list)
        example['sentence'] = text
    if 'sentence' in example:
        example = tokenizer(example['sentence'], max_seq_len=max_seq_length)
    elif 'sentence1' in example:
        example = tokenizer(
            example['sentence1'],
            text_pair=example['sentence2'],
            max_seq_len=max_seq_length)
    if not is_test:
        return example['input_ids'], example['token_type_ids'], label
    else:
        return example['input_ids'], example['token_type_ids']


@paddle.no_grad()
def evaluate(outputs, metric, data_loader):
    metric.reset()
    for i, batch in enumerate(data_loader):
        input_ids, segment_ids, labels = batch
        logits = paddle.to_tensor(outputs[i][0])
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    print("acc: %s, " % res, end='')


class Predictor(object):
    def __init__(self, args):
        inference_backend = InferBackend(
            args.model_path,
            batch_size=args.batch_size,
            device=args.device,
            use_int8=args.int8,
            collect_shape=args.collect_shape,
            num_threads=args.num_threads)
        if args.collect_shape:
            min_batch_size, max_batch_size, opt_batch_size = 1, 32, 32
            min_seq_len, max_seq_len, opt_seq_len = 1, 128, 32
            batches = [
                [
                    np.zeros(
                        [min_batch_size, min_seq_len], dtype="int64"), np.zeros(
                            [min_batch_size, min_seq_len], dtype="int64")
                ],
                [
                    np.zeros(
                        [max_batch_size, max_seq_len], dtype="int64"), np.zeros(
                            [max_batch_size, max_seq_len], dtype="int64")
                ],
                [
                    np.zeros(
                        [opt_batch_size, opt_seq_len], dtype="int64"), np.zeros(
                            [opt_batch_size, opt_seq_len], dtype="int64")
                ],
            ]
            for batch in batches:
                inference_backend.infer(batch)
            print(
                "collect_shape finished, please close collect_shape and restart."
            )
            exit(0)

    def predict_batch(self, data):
        return self.inference_backend.infer(data)

    def predict(self, dataset, tokenizer, batchify_fn, args):
        paddle.disable_static()
        batches = [
            dataset[idx:idx + args.batch_size]
            for idx in range(0, len(dataset), args.batch_size)
        ]

        if args.perf:
            for i, batch in enumerate(batches):
                input_ids, segment_ids, label = batchify_fn(batch)
                output = self.predict_batch([input_ids, segment_ids])
                if i > args.perf_warmup_steps:
                    break
            times = []
            for batch in batches:
                input_ids, segment_ids, _ = batchify_fn(batch)
                time1 = time.time()
                output = self.predict_batch([input_ids, segment_ids])
                times.append(time.time() - time1)

            print("task name: %s, mean time: %s, std time: %s" %
                  (args.task_name, np.mean(times) * 1000, np.std(times) * 1000))

        else:
            metric = METRIC_CLASSES[args.task_name]()
            metric.reset()
            for i, batch in enumerate(batches):
                input_ids, segment_ids, label = batchify_fn(batch)
                output = self.predict_batch([input_ids, segment_ids])
                correct = metric.compute(
                    paddle.to_tensor(output), paddle.to_tensor(label))
                metric.update(correct)

            res = metric.accumulate()
            print("task name: %s, acc: %s, " % (args.task_name, res), end='')


def main():
    paddle.seed(42)
    args = parse_args()

    args.task_name = args.task_name.lower()

    predictor = Predictor(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    dev_ds = load_dataset('clue', args.task_name, splits='dev')

    trans_func = partial(
        convert_example,
        label_list=dev_ds.label_list,
        tokenizer=tokenizer,
        is_test=False)
    dev_ds = dev_ds.map(trans_func, lazy=False)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
        Stack(dtype="int64" if dev_ds.label_list else "float32")  # label
    ): fn(samples)
    outputs = predictor.predict(dev_ds, tokenizer, batchify_fn, args)


if __name__ == "__main__":
    main()
