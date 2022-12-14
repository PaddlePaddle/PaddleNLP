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
import distutils.util
import os

import fastdeploy as fd
import numpy as np

from paddlenlp.transformers import AutoTokenizer


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="The directory of model.")
    parser.add_argument("--vocab_path", type=str, default="", help="The path of tokenizer vocab.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["gpu", "cpu"],
        help="Type of inference device, support 'cpu' or 'gpu'.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="paddle",
        choices=["onnx_runtime", "paddle", "openvino", "tensorrt", "paddle_tensorrt"],
        help="The inference runtime backend.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size of data.")
    parser.add_argument("--max_length", type=int, default=128, help="The max length of sequence.")
    parser.add_argument("--log_interval", type=int, default=10, help="The interval of logging.")
    parser.add_argument("--use_fp16", type=distutils.util.strtobool, default=False, help="Wheter to use FP16 mode")
    parser.add_argument(
        "--use_fast",
        type=distutils.util.strtobool,
        default=True,
        help="Whether to use fast_tokenizer to accelarate the tokenization.",
    )
    return parser.parse_args()


def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


class ErnieForSequenceClassificationPredictor(object):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=args.use_fast)
        self.runtime = self.create_fd_runtime(args)
        self.batch_size = args.batch_size
        self.max_length = args.max_length

    def create_fd_runtime(self, args):
        option = fd.RuntimeOption()
        model_path = os.path.join(args.model_dir, "infer.pdmodel")
        params_path = os.path.join(args.model_dir, "infer.pdiparams")
        option.set_model_path(model_path, params_path)
        if args.device == "cpu":
            option.use_cpu()
        else:
            option.use_gpu()
        if args.backend == "paddle":
            option.use_paddle_infer_backend()
        elif args.backend == "onnx_runtime":
            option.use_ort_backend()
        elif args.backend == "openvino":
            option.use_openvino_backend()
        else:
            option.use_trt_backend()
            if args.backend == "paddle_tensorrt":
                option.enable_paddle_to_trt()
                option.enable_paddle_trt_collect_shape()
            trt_file = os.path.join(args.model_dir, "infer.trt")
            option.set_trt_input_shape(
                "input_ids",
                min_shape=[1, 1],
                opt_shape=[args.batch_size, args.max_length],
                max_shape=[args.batch_size, args.max_length],
            )
            option.set_trt_input_shape(
                "token_type_ids",
                min_shape=[1, 1],
                opt_shape=[args.batch_size, args.max_length],
                max_shape=[args.batch_size, args.max_length],
            )
            if args.use_fp16:
                option.enable_trt_fp16()
                trt_file = trt_file + ".fp16"
            option.set_trt_cache_file(trt_file)
        return fd.Runtime(option)

    def preprocess(self, data):
        data = self.tokenizer(data, max_length=self.max_length, padding=True, truncation=True)
        input_ids_name = self.runtime.get_input_info(0).name
        token_type_ids_name = self.runtime.get_input_info(1).name
        input_map = {
            input_ids_name: np.array(data["input_ids"], dtype="int64"),
            token_type_ids_name: np.array(data["token_type_ids"], dtype="int64"),
        }
        return input_map

    def infer(self, input_map):
        results = self.runtime.infer(input_map)
        return results

    def postprocess(self, infer_data):
        logits = np.array(infer_data[0])
        max_value = np.max(logits, axis=1, keepdims=True)
        exp_data = np.exp(logits - max_value)
        probs = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        out_dict = {"label": probs.argmax(axis=-1), "confidence": probs.max(axis=-1)}
        return out_dict

    def predict(self, data):
        input_map = self.preprocess(data)
        infer_result = self.infer(input_map)
        output = self.postprocess(infer_result)
        return output


def seq_cls_print_ret(infer_result, input_data):
    label_list = [
        "news_story",
        "news_culture",
        "news_entertainment",
        "news_sports",
        "news_finance",
        "news_house",
        "news_car",
        "news_edu",
        "news_tech",
        "news_military",
        "news_travel",
        "news_world",
        "news_stock",
        "news_agriculture",
        "news_game",
    ]
    label = infer_result["label"].tolist()
    confidence = infer_result["confidence"].tolist()
    for i, ret in enumerate(label):
        print("input data:", input_data[i])
        print("seq cls result:")
        print("label:", label_list[label[i]], "  confidence:", confidence[i])
        print("-----------------------------")


if __name__ == "__main__":
    args = parse_arguments()
    predictor = ErnieForSequenceClassificationPredictor(args)
    data = ["未来自动驾驶真的会让酒驾和疲劳驾驶成历史吗？", "黄磊接受华少快问快答，不光智商逆天，情商也不逊黄渤"]
    batch_data = batchfy_text(data, args.batch_size)
    for data in batch_data:
        outputs = predictor.predict(data)
        seq_cls_print_ret(outputs, data)
