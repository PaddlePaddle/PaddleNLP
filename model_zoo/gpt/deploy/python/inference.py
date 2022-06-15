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
import os
from functools import partial

import numpy as np
import paddle
from paddle import inference
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import GPTForGreedyGeneration, GPTChineseTokenizer, GPTTokenizer

MODEL_CLASSES = {
    "gpt-cn": (GPTForGreedyGeneration, GPTChineseTokenizer),
    "gpt": (GPTForGreedyGeneration, GPTTokenizer),
}


def parse_args():
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--model_type", default=None, type=str, required=True, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_path", default=None, type=str, required=True, help="The path prefix of inference model to be used.")
    parser.add_argument("--select_device", default="gpu", choices=["gpu", "cpu", "xpu"], help="Device selected for inference.")
    # yapf: enable

    args = parser.parse_args()
    return args


class Predictor(object):

    def __init__(self, predictor, input_handles, output_handles):
        self.predictor = predictor
        self.input_handles = input_handles
        self.output_handles = output_handles

    @classmethod
    def create_predictor(cls, args):
        config = paddle.inference.Config(args.model_path + ".pdmodel",
                                         args.model_path + ".pdiparams")
        if args.select_device == "gpu":
            # Set GPU configs accordingly
            config.enable_use_gpu(100, 0)
        elif args.select_device == "cpu":
            # Set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
        elif args.select_device == "xpu":
            # Set XPU configs accordingly
            config.enable_xpu(100)
        config.switch_use_feed_fetch_ops(False)
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
            input_handle.copy_from_cpu(input_field.numpy(
            ) if isinstance(input_field, paddle.Tensor) else input_field)
        self.predictor.run()
        output = [
            output_handle.copy_to_cpu() for output_handle in self.output_handles
        ]
        return output

    def predict(self, dataset, batch_size=1):
        outputs = []
        for data in dataset:
            output = self.predict_batch(data)
            outputs.append(output)
        return outputs


def main():
    args = parse_args()
    predictor = Predictor.create_predictor(args)
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(os.path.dirname(
        args.model_path))
    if args.model_type == "gpt":
        ds = [
            "Question: Who is the CEO of Apple? Answer:",
            "Question: Who is the CEO of Facebook? Answer:",
            "Question: How tall is the highest peak in the world? Answer:",
            "Question: Who is the president of the united states? Answer:",
            "Question: Where is the capital of France? Answer:",
            "Question: What is the largest animal in the ocean? Answer:",
            "Question: Who is the chancellor of Germany? Answer:",
        ]
    elif args.model_type == "gpt-cn":
        ds = [
            "问题：苹果的CEO是谁? 答案：",
            "问题：中国的首都是哪里？答案：",
            "问题：世界上最高的山峰是? 答案：",
        ]

    dataset = [[
        np.array(tokenizer(text)["input_ids"]).astype("int64").reshape([1, -1])
    ] for text in ds]
    outs = predictor.predict(dataset)
    for res in outs:
        res_ids = list(res[0].reshape([-1]))
        res_ids = [int(x) for x in res_ids]
        print(tokenizer.convert_ids_to_string(res_ids))


if __name__ == "__main__":
    main()
