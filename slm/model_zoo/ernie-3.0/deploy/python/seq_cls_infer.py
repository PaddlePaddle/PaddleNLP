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
    parser.add_argument("--model_prefix", type=str, default="model", help="The model and params file prefix.")
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
    return parser.parse_args()


def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


class Predictor(object):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        self.runtime = self.create_fd_runtime(args)
        self.batch_size = args.batch_size
        self.max_length = args.max_length

    def create_fd_runtime(self, args):
        option = fd.RuntimeOption()
        model_path = os.path.join(args.model_dir, args.model_prefix + ".pdmodel")
        params_path = os.path.join(args.model_dir, args.model_prefix + ".pdiparams")
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
            trt_file = os.path.join(args.model_dir, "model.trt")
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

    def preprocess(self, text, text_pair):
        data = self.tokenizer(text, text_pair, max_length=self.max_length, padding=True, truncation=True)
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

    def predict(self, texts, texts_pair=None):
        input_map = self.preprocess(texts, texts_pair)
        infer_result = self.infer(input_map)
        output = self.postprocess(infer_result)
        return output


if __name__ == "__main__":
    args = parse_arguments()
    predictor = Predictor(args)
    texts_ds = ["花呗收款额度限制", "花呗支持高铁票支付吗"]
    texts_pair_ds = ["收钱码，对花呗支付的金额有限制吗", "为什么友付宝不支持花呗付款"]
    batch_texts = batchfy_text(texts_ds, args.batch_size)
    batch_texts_pair = batchfy_text(texts_pair_ds, args.batch_size)

    for bs, (texts, texts_pair) in enumerate(zip(batch_texts, batch_texts_pair)):
        outputs = predictor.predict(texts, texts_pair)
        for i, (sentence1, sentence2) in enumerate(zip(texts, texts_pair)):
            print(
                f"Batch id:{bs}, example id:{i}, sentence1:{sentence1}, sentence2:{sentence2}, label:{outputs['label'][i]}, similarity:{outputs['confidence'][i]:.4f}"
            )
