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

import numpy as np
import paddle
from paddle import inference
from paddlenlp.datasets import load_dataset

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True, default="./export/", help="The directory to static model.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
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
        config.delete_pass("embedding_eltwise_layernorm_fuse_pass")
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
        labels = []
        for pred in preds:
            # drop the concated CLS and SEP token label
            pred = pred[1:-1]
            label = [label_map[i] for i in pred]
            labels.append(label)
        return labels


if __name__ == "__main__":
    # Define predictor to do prediction.
    predictor = Predictor(args.model_dir, args.device, args.batch_size,
                          args.cpu_threads, args.enable_mkldnn)

    predict_ds = load_dataset('msra_ner', splits=('test'))
    texts = ["".join(example["tokens"]) for example in predict_ds]
    batches = [
        texts[idx:idx + args.batch_size]
        for idx in range(0, len(texts), args.batch_size)
    ]
    label_map = dict(enumerate(predict_ds.label_list))

    results = []
    for batch_data in batches:
        labels = predictor.predict(batch_data, label_map)
        results.extend(labels)

    for idx, text in enumerate(texts):
        seq_len = len(text)
        label = results[idx][:seq_len]
        print(text, " : ", " ".join(label))
