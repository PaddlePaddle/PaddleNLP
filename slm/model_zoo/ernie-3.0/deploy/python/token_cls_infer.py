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


class ErnieForTokenClassificationPredictor:
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        self.predictor = self.create_predictor(args)
        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.label_names = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]

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

    def preprocess(self, texts):
        is_split_into_words = isinstance(texts[0], list)
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            is_split_into_words=is_split_into_words,
            return_tensors="np",
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

    def postprocess(self, infer_data, input_data):
        result = np.array(infer_data)
        tokens_label = result.argmax(axis=-1).tolist()
        value = []
        for batch, token_label in enumerate(tokens_label):
            start = -1
            label_name = ""
            items = []
            for i, label in enumerate(token_label):
                label_str = self.label_names[label]
                if (label_str == "O" or "B-" in label_str) and start >= 0:
                    entity = input_data[batch][start : i - 1]
                    if isinstance(entity, list):
                        entity = "".join(entity)
                    if len(entity) == 0:
                        break
                    items.append(
                        {
                            "pos": [start, i - 2],
                            "entity": entity,
                            "label": label_name,
                        }
                    )
                    start = -1
                if "B-" in label_str:
                    start = i - 1
                    label_name = label_str[2:]
            value.append(items)

        return {"value": value, "tokens_label": tokens_label}

    def predict(self, texts):
        input_map = self.preprocess(texts)
        infer_result = self.infer(input_map)
        return self.postprocess(infer_result, texts)


def token_cls_print_ret(infer_result, input_data):
    rets = infer_result["value"]
    for i, ret in enumerate(rets):
        print("input data:", input_data[i])
        print("The model detects all entities:")
        for item in ret:
            print("entity:", item["entity"], "  label:", item["label"], "  pos:", item["pos"])
        print("-----------------------------")


if __name__ == "__main__":
    args = parse_arguments()
    predictor = ErnieForTokenClassificationPredictor(args)
    texts = ["北京的涮肉，重庆的火锅，成都的小吃都是极具特色的美食。", "乔丹、科比、詹姆斯和姚明都是篮球界的标志性人物。"]
    batch_data = batchfy_text(texts, args.batch_size)

    for data in batch_data:
        outputs = predictor.predict(data)
        token_cls_print_ret(outputs, data)
