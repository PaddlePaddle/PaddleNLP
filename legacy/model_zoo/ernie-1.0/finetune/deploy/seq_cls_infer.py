# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.datasets import load_dataset
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
    parser.add_argument("--cpu_threads", type=int, default=1, help="Number of threads to predict when using cpu.")
    parser.add_argument("--device_id", type=int, default=0, help="Select which gpu device to train model.")
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
            option.set_cpu_thread_num(args.cpu_threads)
        else:
            option.use_gpu(args.device_id)
        if args.backend == "paddle":
            option.use_paddle_infer_backend()
        elif args.backend == "onnx_runtime":
            option.use_ort_backend()
        elif args.backend == "openvino":
            option.use_openvino_backend()
        else:
            option.use_trt_backend()
            if args.backend == "paddle_tensorrt":
                option.use_paddle_infer_backend()
                option.paddle_infer_option.collect_trt_shape = True
                option.paddle_infer_option.enable_trt = True
            trt_file = os.path.join(args.model_dir, "model.trt")
            option.trt_option.set_shape(
                "input_ids", [1, 1], [args.batch_size, args.max_length], [args.batch_size, args.max_length]
            )
            option.trt_option.set_shape(
                "token_type_ids", [1, 1], [args.batch_size, args.max_length], [args.batch_size, args.max_length]
            )
            if args.use_fp16:
                option.trt_option.enable_fp16 = True
                trt_file = trt_file + ".fp16"
            option.trt_option.serialize_file = trt_file
        return fd.Runtime(option)

    def preprocess(self, text):
        data = self.tokenizer(text, max_length=self.max_length, padding=True, truncation=True)
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
        out_dict = {"label": probs.argmax(axis=-1), "confidence": probs}
        return out_dict

    def predict(self, texts):
        input_map = self.preprocess(texts)
        infer_result = self.infer(input_map)
        output = self.postprocess(infer_result)
        return output


if __name__ == "__main__":
    args = parse_arguments()
    predictor = Predictor(args)
    test_ds = load_dataset("chnsenticorp", splits=["test"])
    texts_ds = [d["text"] for d in test_ds]
    label_map = {0: "negative", 1: "positive"}
    batch_texts = batchfy_text(texts_ds, args.batch_size)
    for bs, texts in enumerate(batch_texts):
        outputs = predictor.predict(texts)
        for i, sentence1 in enumerate(texts):
            print(
                f"Batch id: {bs}, example id: {i}, sentence: {sentence1}, "
                f"label: {label_map[outputs['label'][i]]}, negative prob: {outputs['confidence'][i][0]:.4f}, "
                f"positive prob: {outputs['confidence'][i][1]:.4f}."
            )
