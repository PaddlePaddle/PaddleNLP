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
import time
from functools import partial
import distutils.util
import numpy as np

import paddle
from paddle import inference
from paddle.metric import Accuracy

from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoTokenizer

from infer_backend import InferBackend


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_name_or_path",
        default="ernie-3.0-medium-zh",
        type=str,
        help="The directory or name of model.",
    )
    parser.add_argument(
        "--model_path",
        default='inference/infer',
        type=str,
        help="The path prefix of inference model to be used.",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu", "xpu"],
        help="Device selected for inference.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for predict.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--perf_warmup_steps",
        default=20,
        type=int,
        help="Warmup steps for performance test.",
    )
    parser.add_argument(
        "--use_trt",
        action='store_true',
        help="Whether to use inference engin TensorRT.",
    )
    parser.add_argument(
        "--perf",
        action='store_true',
        help="Whether to test performance.",
    )
    parser.add_argument(
        "--int8",
        action='store_true',
        help="Whether to use int8 inference.",
    )
    parser.add_argument(
        "--faster_tokenizer",
        action='store_true',
        help="Whether to use FasterTokenizer.",
    )
    parser.add_argument("--use_onnxruntime",
                        type=distutils.util.strtobool,
                        default=False,
                        help="Use onnxruntime to infer or not.")
    parser.add_argument(
        "--num_threads",
        default=1,
        type=int,
        help="Set num_threads.",
    )
    parser.add_argument(
        "--epochs",
        default=1,
        type=int,
        help="Set epochs.",
    )
    parser.add_argument(
        "--fp16",
        action='store_true',
        help="Whether to use int16 inference.",
    )
    parser.add_argument(
        "--use_inference",
        action='store_true',
        help="use_inference for inference.",
    )
    parser.add_argument(
        "--collect_shape",
        action='store_true',
        help="Whether to collect shape info.",
    )
    args = parser.parse_args()
    return args


def convert_example(example,
                    label_list,
                    tokenizer=None,
                    is_test=False,
                    max_seq_length=512,
                    **kwargs):
    """convert a glue example into necessary features"""
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        example['label'] = np.array(example["label"], dtype="int64")
        label = example['label']
    if tokenizer is None:
        return example
    if 'sentence' in example:
        example = tokenizer(example['sentence'], max_seq_len=max_seq_length)
    elif 'sentence1' in example:
        example = tokenizer(example['sentence1'],
                            text_pair=example['sentence2'],
                            max_seq_len=max_seq_length)
    if not is_test:
        return example['input_ids'], example['token_type_ids'], label
    else:
        return example['input_ids'], example['token_type_ids']


class Predictor(object):

    def __init__(self, args):
        self.inference_backend = InferBackend(args.model_path,
                                              batch_size=args.batch_size,
                                              device=args.device,
                                              use_int8=args.int8,
                                              use_fp16=args.fp16,
                                              collect_shape=args.collect_shape,
                                              num_threads=args.num_threads,
                                              use_inference=args.use_inference,
                                              use_trt=args.use_trt)
        if args.collect_shape:
            min_batch_size, max_batch_size, opt_batch_size = 1, args.batch_size, args.batch_size
            min_seq_len, max_seq_len, opt_seq_len = 2, 128, 32
            batches = [
                [
                    np.zeros([min_batch_size, min_seq_len], dtype="int64"),
                    np.zeros([min_batch_size, min_seq_len], dtype="int64")
                ],
                [
                    np.zeros([max_batch_size, max_seq_len], dtype="int64"),
                    np.zeros([max_batch_size, max_seq_len], dtype="int64")
                ],
                [
                    np.zeros([opt_batch_size, opt_seq_len], dtype="int64"),
                    np.zeros([opt_batch_size, opt_seq_len], dtype="int64")
                ],
            ]
            for batch in batches:
                self.inference_backend.infer(batch)
            print(
                "collect_shape finished, please close collect_shape and restart."
            )
            exit(0)

    def predict_batch(self, data):
        return self.inference_backend.infer(data)

    def convert_predict_batch(self, args, data, tokenizer, label_list):
        examples = []
        for example in data:
            example = convert_example(example,
                                      label_list,
                                      tokenizer,
                                      max_seq_length=args.max_seq_length)
            examples.append(example)
        if tokenizer is None:
            labels = [example["label"] for example in examples]
            if "sentence" in examples[0]:
                examples = [example["sentence"] for example in examples]
            else:
                examples = [(example["sentence1"], example["sentence2"])
                            for example in examples]
        return examples, labels

    def predict(self, dataset, tokenizer, args):
        paddle.disable_static()
        batches = [
            dataset[idx:idx + args.batch_size]
            for idx in range(0, len(dataset), args.batch_size)
        ]
        if args.perf:
            for i, batch in enumerate(batches):
                examples, _ = self.convert_predict_batch(
                    args, batch, None, dataset.label_list)
                encodings = tokenizer(examples,
                                      padding=True,
                                      max_length=args.max_seq_length,
                                      truncation=True,
                                      return_tensors="np")
                input_ids = encodings['input_ids']
                segment_ids = encodings['token_type_ids']
                output = self.predict_batch([input_ids, segment_ids])
                output_label = np.argmax(output, -1)
                if i > args.perf_warmup_steps:
                    break

            time1 = time.time()
            nums = 0
            for j in range(args.epochs):
                for batch in batches:
                    nums = nums + len(batch)
                    examples, _ = self.convert_predict_batch(
                        args, batch, None, dataset.label_list)
                    encodings = tokenizer(examples,
                                          padding=True,
                                          max_length=args.max_seq_length,
                                          truncation=True,
                                          return_tensors="np")
                    input_ids = encodings['input_ids']
                    segment_ids = encodings['token_type_ids']
                    output = self.predict_batch([input_ids, segment_ids])
                    output_label = np.argmax(output, -1)
            times_total = time.time() - time1
            log_info = "model name: %s, thread num: %s, nums: %s, int8: %s, fp16: %s, times_total: %s, QPS: %s seq/s, latency: %s ms" % (
                args.model_path, args.num_threads, nums, args.int8, args.fp16,
                times_total, nums / times_total, times_total * 1000 / nums)
            print(log_info)
            with open("iflytek_cpu.txt", 'a+') as f:
                f.write(log_info + "\n")
        else:
            metric = Accuracy()
            metric.reset()
            for i, batch in enumerate(batches):
                # Data Preprocess
                examples, labels = self.convert_predict_batch(
                    args, batch, None, dataset.label_list)
                encodings = tokenizer(examples,
                                      padding=True,
                                      max_length=args.max_seq_length,
                                      truncation=True,
                                      return_tensors="np")
                input_ids = encodings['input_ids']
                segment_ids = encodings['token_type_ids']
                # Predict
                output = self.predict_batch([input_ids, segment_ids])
                # Compute metric, for eval
                correct = metric.compute(paddle.to_tensor(output),
                                         paddle.to_tensor(labels))
                metric.update(correct)

            res = metric.accumulate()
            log_info = "model name: %s, acc: %s, " % (args.model_path, res)
            print(log_info)
            with open("iflytek_cpu.txt", 'a+') as f:
                f.write(log_info + "\n")


def main():
    paddle.seed(42)
    args = parse_args()

    args.task_name = "iflytek"

    predictor = Predictor(args)

    dev_ds = load_dataset('clue', args.task_name, splits='dev')
    # Decalare a tokenizer by model name
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              use_faster=args.faster_tokenizer)
    print("tokenizer type: ", type(tokenizer))
    # predict
    outputs = predictor.predict(dev_ds, tokenizer, args)


if __name__ == "__main__":
    main()
