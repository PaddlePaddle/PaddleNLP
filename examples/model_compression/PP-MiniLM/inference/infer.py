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
import sys
from functools import partial
import numpy as np

import paddle
from paddle import inference
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad

sys.path.append("../")
from data import convert_example, METRIC_CLASSES, MODEL_CLASSES


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default='afqmc',
        type=str,
        help="The name of the task to perform predict, selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()), )
    parser.add_argument(
        "--model_type",
        default='ernie',
        type=str,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default='ppminilm-6l-768h',
        type=str,
        help="The directory or name of model.", )
    parser.add_argument(
        "--model_path",
        default='./quant_models/model',
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
        "--use_trt",
        action='store_true',
        help="Whether to use inference engin TensorRT.", )
    parser.add_argument(
        "--perf",
        action='store_true',
        help="Whether to test performance.", )
    parser.add_argument(
        "--collect_shape",
        action='store_true',
        help="Whether collect shape range info.", )
    parser.add_argument(
        "--int8",
        action='store_true',
        help="Whether to use int8 inference.", )
    args = parser.parse_args()
    return args


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
    def __init__(self, predictor, input_handles, output_handles):
        self.predictor = predictor
        self.input_handles = input_handles
        self.output_handles = output_handles

    @classmethod
    def create_predictor(cls, args):
        config = paddle.inference.Config(args.model_path + ".pdmodel",
                                         args.model_path + ".pdiparams")
        if args.device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)
            cls.device = paddle.set_device("gpu")
        elif args.device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
            cls.device = paddle.set_device("cpu")
        elif args.device == "xpu":
            # set XPU configs accordingly
            config.enable_xpu(100)
        if args.use_trt:
            if args.int8:
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=inference.PrecisionType.Int8,
                    max_batch_size=args.batch_size,
                    min_subgraph_size=5,
                    use_static=False,
                    use_calib_mode=False)
            else:
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=inference.PrecisionType.Float32,
                    max_batch_size=args.batch_size,
                    min_subgraph_size=5,
                    use_static=False,
                    use_calib_mode=False)
            print("Enable TensorRT is: {}".format(
                config.tensorrt_engine_enabled()))
        if args.collect_shape:
            config.collect_shape_range_info(
                os.path.join(
                    os.path.dirname(args.model_path), args.task_name +
                    '_shape_range_info.pbtxt'))
        else:
            config.enable_tuned_tensorrt_dynamic_shape(
                os.path.join(
                    os.path.dirname(args.model_path),
                    args.task_name + "_shape_range_info.pbtxt"), True)
        predictor = paddle.inference.create_predictor(config)
        input_handles = [
            predictor.get_input_handle(name)
            for name in predictor.get_input_names()
        ]
        output_handles = [
            predictor.get_output_handle(name)
            for name in predictor.get_output_names()
        ]

        return cls(predictor, input_handles, output_handles)

    def predict_batch(self, data):
        for input_field, input_handle in zip(data, self.input_handles):
            input_handle.copy_from_cpu(input_field.numpy() if isinstance(
                input_field, paddle.Tensor) else input_field)
        self.predictor.run()
        output = [
            output_handle.copy_to_cpu() for output_handle in self.output_handles
        ]

        return output

    def predict(self, dataset, collate_fn, args, batch_size=1):
        metric = METRIC_CLASSES[args.task_name]()
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=False)
        data_loader = paddle.io.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=0,
            return_list=True)
        outputs = []
        metric.reset()
        for i, data in enumerate(data_loader):
            if len(data) == 2:
                output = self.predict_batch(data)
            else:
                output = self.predict_batch([data[0], data[1]])
                logits = paddle.to_tensor(output)
                correct = metric.compute(logits, data[2])
                metric.update(correct)
            outputs.append(output)
        if len(data) > 2:
            res = metric.accumulate()
            print("task name: %s, acc: %s, " % (args.task_name, res), end='')

        return outputs

    def predict_perf(self, dataset, collate_fn, args, batch_size=1):
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=False)
        data_loader = paddle.io.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=0,
            return_list=True)
        time1 = time.time()
        for i, data in enumerate(data_loader):
            if i < args.perf_warmup_steps:  # skip warmup steps.
                continue
            output = self.predict_batch([data[0], data[1]])
            logits = paddle.to_tensor(output)

        print("time: ", time.time() - time1)


def main():
    paddle.seed(42)
    args = parse_args()

    args.task_name = args.task_name.lower()
    args.model_type = args.model_type.lower()

    predictor = Predictor.create_predictor(args)

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    dev_ds = load_dataset('clue', args.task_name, splits='dev')
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=dev_ds.label_list,
        max_seq_length=args.max_seq_length,
        is_test=False)

    dev_ds = dev_ds.map(trans_func, lazy=True)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64" if dev_ds.label_list else "float32")  # label
    ): fn(samples)
    if args.perf:
        outputs = predictor.predict_perf(
            dev_ds,
            batch_size=args.batch_size,
            collate_fn=batchify_fn,
            args=args)
    else:
        outputs = predictor.predict(
            dev_ds,
            batch_size=args.batch_size,
            collate_fn=batchify_fn,
            args=args)


if __name__ == "__main__":
    main()
