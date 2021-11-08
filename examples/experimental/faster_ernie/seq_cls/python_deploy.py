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

import numpy as np
import paddle
import paddlenlp as ppnlp
from scipy.special import softmax
from paddle import inference
from paddlenlp.datasets import load_dataset

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True, default="./export/", help="The directory to static model.")
parser.add_argument("--max_seq_length", type=int, default=128, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", type=int, default=3, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument('--use_tensorrt', type=eval, default=False, choices=[True, False], help='Enable to use tensorrt to speed up.')
parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "int8"], help='The tensorrt precision.')
parser.add_argument('--cpu_threads', type=int, default=10, help='Number of threads to predict when using cpu.')
parser.add_argument('--enable_mkldnn', type=eval, default=False, choices=[True, False], help='Enable to use mkldnn to speed up when using cpu.')
parser.add_argument("--save_log_path", type=str, default="./log_output/", help="The file path to save log.")
args = parser.parse_args()
# yapf: enable


class Predictor(object):
    def __init__(self,
                 model_dir,
                 device="gpu",
                 batch_size=32,
                 use_tensorrt=False,
                 precision="fp32",
                 cpu_threads=10,
                 enable_mkldnn=False):
        self.batch_size = batch_size

        model_file = os.path.join(model_dir, "inference.pdmodel")
        params_file = os.path.join(model_dir, "inference.pdiparams")
        if not os.path.exists(model_file):
            raise ValueError("The model file {} is not found.".format(
                model_file))
        if not os.path.exists(params_file):
            raise ValueError("The params file {} is not found.".format(
                params_file))
        config = paddle.inference.Config(model_file, params_file)

        if device == "gpu":
            # set GPU configs accordingly
            # such as intialize the gpu memory, enable tensorrt
            config.enable_use_gpu(100, 0)
            precision_map = {
                "fp16": inference.PrecisionType.Half,
                "fp32": inference.PrecisionType.Float32,
                "int8": inference.PrecisionType.Int8
            }
            precision_mode = precision_map[precision]

            if use_tensorrt:
                config.enable_tensorrt_engine(
                    max_batch_size=batch_size,
                    min_subgraph_size=30,
                    precision_mode=precision_mode)
        elif device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
            if enable_mkldnn:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
            config.set_cpu_math_library_num_threads(cpu_threads)
        elif device == "xpu":
            # set XPU configs accordingly
            config.enable_xpu(100)

        config.switch_use_feed_fetch_ops(False)
        self.predictor = paddle.inference.create_predictor(config)
        self.input_handle = self.predictor.get_input_handle(
            self.predictor.get_input_names()[0])
        self.output_handles = [
            self.predictor.get_output_handle(name)
            for name in self.predictor.get_output_names()
        ]

    def predict(self, data, label_map):
        self.input_handle.copy_from_cpu(data)
        self.predictor.run()
        logits = self.output_handles[0].copy_to_cpu()
        preds = self.output_handles[1].copy_to_cpu()
        labels = [label_map[pred] for pred in preds]
        return labels


if __name__ == "__main__":
    # Define predictor to do prediction.
    predictor = Predictor(args.model_dir, args.device, args.batch_size,
                          args.use_tensorrt, args.precision, args.cpu_threads,
                          args.enable_mkldnn)

    test_ds = load_dataset("chnsenticorp", splits=["test"])
    label_map = {0: "negative", 1: "positive"}
    data = [example["text"] for example in test_ds]
    batches = [
        data[idx:idx + args.batch_size]
        for idx in range(0, len(data), args.batch_size)
    ]
    results = []
    for batch in batches:
        labels = predictor.predict(batch, label_map=label_map)
        results.extend(labels)

    file_path = "results.txt"
    print(
        "The results have been saved to the file: %s, some results are shown as below: "
        % file_path)
    with open(file_path, "w", encoding="utf8") as fout:
        for idx, text in enumerate(data):
            if idx < 10:
                print(text, " : ", results[idx])
            fout.write(text + " : " + results[idx] + "\n")
