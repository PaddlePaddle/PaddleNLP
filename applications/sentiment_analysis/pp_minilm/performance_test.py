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
import time
import paddle
from paddle import inference
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import PPMiniLMTokenizer
from paddlenlp.metrics import AccuracyAndF1

from data import read, load_dict, convert_example_to_feature


class Predictor(object):

    def __init__(self, args):
        self.predictor, self.input_handles, self.output_handles = self.create_predictor(
            args)

    def create_predictor(self, args):
        config = paddle.inference.Config(args.model_path + ".pdmodel",
                                         args.model_path + ".pdiparams")
        if args.device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)
            paddle.set_device("gpu")
        elif args.device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
            paddle.set_device("cpu")
        elif args.device == "xpu":
            # set XPU configs accordingly
            config.enable_xpu(100)
        if args.use_tensorrt:
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
                os.path.join(os.path.dirname(args.model_path),
                             'collect_shape_range_info.pbtxt'))
        else:
            config.enable_tuned_tensorrt_dynamic_shape(
                os.path.join(os.path.dirname(args.model_path),
                             "collect_shape_range_info.pbtxt"), True)

        predictor = paddle.inference.create_predictor(config)
        input_handles = [
            predictor.get_input_handle(name)
            for name in predictor.get_input_names()
        ]
        output_handles = [
            predictor.get_output_handle(name)
            for name in predictor.get_output_names()
        ]

        return predictor, input_handles, output_handles

    def predict_batch(self, data):
        for input_field, input_handle in zip(data, self.input_handles):
            input_handle.copy_from_cpu(input_field.numpy(
            ) if isinstance(input_field, paddle.Tensor) else input_field)
        self.predictor.run()
        output = [
            output_handle.copy_to_cpu() for output_handle in self.output_handles
        ]

        return output

    def predict(self, data_loader, metric):

        outputs = []
        metric.reset()
        for i, data in enumerate(data_loader):
            output = self.predict_batch([data[0], data[1]])
            logits = paddle.to_tensor(output).squeeze(0)
            correct = metric.compute(logits, paddle.to_tensor(data[3]))
            metric.update(correct)
            outputs.append(output)

        accuracy, precision, recall, F1, _ = metric.accumulate()
        return outputs, accuracy, precision, recall, F1

    def predict_perf(self, args, data_loader):
        start_time = time.time()
        for i, data in enumerate(data_loader):
            if i < args.perf_warmup_steps:  # skip warmup steps.
                continue
            output = self.predict_batch([data[0], data[1]])
            logits = paddle.to_tensor(output)

        used_time = time.time() - start_time
        return used_time


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default=None, help="The name of base model.")
    parser.add_argument("--model_path", default='./checkpoints/quant/infer', type=str, required=True, help="The path prefix of inference model to be used.")
    parser.add_argument('--test_path', type=str, default=None, help="The path of test set.")
    parser.add_argument("--label_path", type=str, default=None, help="The path of label dict.")
    parser.add_argument("--num_epochs", type=int, default=0, help="Number of epoches for training.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", default=256, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--perf_warmup_steps", default=1, type=int, help="Warmup steps for performance test.")
    parser.add_argument("--use_tensorrt", action='store_true', help="Whether to use inference engin TensorRT.")
    parser.add_argument("--eval", action='store_true', help="Whether to test performance.")
    parser.add_argument("--collect_shape", action='store_true', help="Whether collect shape range info.")
    parser.add_argument("--int8", action='store_true', help="Whether to use int8 inference.")
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu", "xpu"], help="Device selected for inference.")

    args = parser.parse_args()
    # yapf: enable

    # set running environnent
    paddle.seed(42)

    label2id, id2label = load_dict(args.label_path)
    test_ds = load_dataset(read, data_path=args.test_path, lazy=False)

    tokenizer = PPMiniLMTokenizer.from_pretrained(args.base_model_name)
    trans_func = partial(convert_example_to_feature,
                         tokenizer=tokenizer,
                         label2id=label2id,
                         max_seq_len=args.max_seq_len,
                         is_test=False)
    test_ds = test_ds.map(trans_func, lazy=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"
            ),  # segment
        Stack(dtype="int64"),  # seq_len
        Stack(dtype="int64")  # label
    ): fn(samples)

    batch_sampler = paddle.io.BatchSampler(test_ds,
                                           batch_size=args.batch_size,
                                           shuffle=False)
    data_loader = paddle.io.DataLoader(dataset=test_ds,
                                       batch_sampler=batch_sampler,
                                       collate_fn=batchify_fn,
                                       num_workers=0,
                                       return_list=True)

    predictor = Predictor(args)

    if args.num_epochs > 0:
        print("start to do performance task.")
        times = []
        for epoch_id in range(1, args.num_epochs + 1):
            used_time = predictor.predict_perf(args, data_loader)
            times.append(used_time)
            print(f"epoch {epoch_id}, used_time: {used_time}")
        print(f"the avg time of {args.num_epochs} epochs is {np.mean(times)}")

    if args.eval:
        print("start to do evaluate task.")
        metric = AccuracyAndF1()
        outputs, accuracy, precision, recall, F1 = predictor.predict(
            data_loader, metric)
        print(
            f"evalute results - accuracy: {accuracy: .5f}, precision: {precision: .5f}, recall: {recall: .5f}, F1: {F1: .5f}"
        )
