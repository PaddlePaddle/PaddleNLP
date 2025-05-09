# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.inference as paddle_infer

from paddlenlp.transformers import AutoTokenizer
from paddlenlp.utils.env import (
    PADDLE_INFERENCE_MODEL_SUFFIX,
    PADDLE_INFERENCE_WEIGHTS_SUFFIX,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="The directory of model.")
    parser.add_argument("--vocab_path", type=str, default="", help="The path of tokenizer vocab.")
    parser.add_argument("--model_prefix", type=str, default="model", help="The model and params file prefix.")
    parser.add_argument("--device", type=str, default="cpu", choices=["gpu", "cpu"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--log_interval", type=int, default=10)
    return parser.parse_args()


def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts.append(texts[batch_start : batch_start + batch_size])
        batch_start += batch_size
    return batch_texts


class Predictor:
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        self.predictor = self.create_predictor(args)
        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()
        self.batch_size = args.batch_size
        self.max_length = args.max_length

    def create_predictor(self, args):
        model_path = os.path.join(args.model_dir, args.model_prefix + f"{PADDLE_INFERENCE_MODEL_SUFFIX}")
        params_path = os.path.join(args.model_dir, args.model_prefix + f"{PADDLE_INFERENCE_WEIGHTS_SUFFIX}")
        config = paddle_infer.Config(model_path, params_path)

        if args.device == "gpu":
            config.enable_use_gpu(100, 0)
        else:
            config.disable_gpu()
        config.switch_use_feed_fetch_ops(False)
        config.enable_memory_optim()
        return paddle_infer.create_predictor(config)

    def preprocess(self, text, text_pair):
        encoded = self.tokenizer(
            text, text_pair, max_length=self.max_length, padding=True, truncation=True, return_tensors="np"
        )
        return {
            "input_ids": encoded["input_ids"].astype("int64"),
            "token_type_ids": encoded["token_type_ids"].astype("int64"),
        }

    def infer(self, input_map):
        input_ids_handle = self.predictor.get_input_handle(self.input_names[0])
        token_type_ids_handle = self.predictor.get_input_handle(self.input_names[1])

        input_ids_handle.copy_from_cpu(input_map["input_ids"])
        token_type_ids_handle.copy_from_cpu(input_map["token_type_ids"])

        self.predictor.run()

        output_handle = self.predictor.get_output_handle(self.output_names[0])
        return output_handle.copy_to_cpu()

    def postprocess(self, logits):
        max_value = np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits - max_value)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        return {"label": np.argmax(probs, axis=1), "confidence": np.max(probs, axis=1)}

    def predict(self, texts, texts_pair=None):
        input_map = self.preprocess(texts, texts_pair)
        logits = self.infer(input_map)
        return self.postprocess(logits)


if __name__ == "__main__":
    args = parse_arguments()
    predictor = Predictor(args)

    texts_ds = ["花呗收款额度限制", "花呗支持高铁票支付吗"]
    texts_pair_ds = ["收钱码，对花呗支付的金额有限制吗", "为什么友付宝不支持花呗付款"]

    batch_texts = batchfy_text(texts_ds, args.batch_size)
    batch_texts_pair = batchfy_text(texts_pair_ds, args.batch_size)

    for bs, (texts, texts_pair) in enumerate(zip(batch_texts, batch_texts_pair)):
        outputs = predictor.predict(texts, texts_pair)
        for i, (s1, s2) in enumerate(zip(texts, texts_pair)):
            print(
                f"Batch {bs}, example {i} | s1: {s1} | s2: {s2} | label: {outputs['label'][i]} | score: {outputs['confidence'][i]:.4f}"
            )
