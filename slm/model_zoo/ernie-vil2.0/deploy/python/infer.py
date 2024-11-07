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
from PIL import Image

from paddlenlp.transformers import ErnieViLProcessor


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="The directory of model.")
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu", "kunlunxin"],
        help="Type of inference device, support 'cpu', 'kunlunxin' or 'gpu'.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="onnx_runtime",
        choices=["onnx_runtime", "paddle", "openvino", "tensorrt", "paddle_tensorrt"],
        help="The inference runtime backend.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size of data.")
    parser.add_argument("--temperature", type=float, default=4.30022621, help="The temperature of the model.")
    parser.add_argument("--max_length", type=int, default=128, help="The max length of sequence.")
    parser.add_argument("--log_interval", type=int, default=10, help="The interval of logging.")
    parser.add_argument("--use_fp16", type=distutils.util.strtobool, default=False, help="Wheter to use FP16 mode")
    parser.add_argument(
        "--encode_type",
        type=str,
        default="text",
        choices=[
            "image",
            "text",
        ],
        help="The encoder type.",
    )
    parser.add_argument(
        "--image_path",
        default="000000039769.jpg",
        type=str,
        help="image_path used for prediction",
    )
    return parser.parse_args()


class ErnieVil2Predictor(object):
    def __init__(self, args):
        self.processor = ErnieViLProcessor.from_pretrained("PaddlePaddle/ernie_vil-2.0-base-zh")
        self.runtime = self.create_fd_runtime(args)
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.encode_type = args.encode_type

    def create_fd_runtime(self, args):
        option = fd.RuntimeOption()
        if args.encode_type == "text":
            model_path = os.path.join(args.model_dir, "get_text_features.pdmodel")
            params_path = os.path.join(args.model_dir, "get_text_features.pdiparams")
        else:
            model_path = os.path.join(args.model_dir, "get_image_features.pdmodel")
            params_path = os.path.join(args.model_dir, "get_image_features.pdiparams")
        option.set_model_path(model_path, params_path)
        if args.device == "kunlunxin":
            option.use_kunlunxin()
            option.use_paddle_lite_backend()
            return fd.Runtime(option)
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
            trt_file = os.path.join(args.model_dir, "{}_infer.trt".format(args.encode_type))
            if args.encode_type == "text":
                option.set_trt_input_shape(
                    "input_ids",
                    min_shape=[1, args.max_length],
                    opt_shape=[args.batch_size, args.max_length],
                    max_shape=[args.batch_size, args.max_length],
                )
            else:
                option.set_trt_input_shape(
                    "pixel_values",
                    min_shape=[1, 3, 224, 224],
                    opt_shape=[args.batch_size, 3, 224, 224],
                    max_shape=[args.batch_size, 3, 224, 224],
                )
            if args.use_fp16:
                option.enable_trt_fp16()
                trt_file = trt_file + ".fp16"
            option.set_trt_cache_file(trt_file)
        return fd.Runtime(option)

    def preprocess(self, inputs):
        if self.encode_type == "text":
            dataset = [np.array([self.processor(text=text)["input_ids"] for text in inputs], dtype="int64")]
        else:
            dataset = [np.array([self.processor(images=image)["pixel_values"][0] for image in inputs])]
        input_map = {}
        for input_field_id, data in enumerate(dataset):
            input_field = self.runtime.get_input_info(input_field_id).name
            input_map[input_field] = data
        return input_map

    def postprocess(self, infer_data):
        logits = np.array(infer_data[0])
        out_dict = {
            "features": logits,
        }
        return out_dict

    def infer(self, input_map):
        results = self.runtime.infer(input_map)
        return results

    def predict(self, inputs):
        input_map = self.preprocess(inputs)
        infer_result = self.infer(input_map)
        output = self.postprocess(infer_result)
        return output


def main():
    args = parse_arguments()
    texts = [
        "猫的照片",
        "狗的照片",
    ]
    args.batch_size = 2
    predictor = ErnieVil2Predictor(args)
    outputs = predictor.predict(texts)
    print(outputs)
    text_feats = outputs["features"]
    image = Image.open(args.image_path)
    args.encode_type = "image"
    args.batch_size = 1
    predictor = ErnieVil2Predictor(args)
    images = [image]
    outputs = predictor.predict(images)
    image_feats = outputs["features"]
    print(image_feats)
    from scipy.special import softmax

    image_feats = image_feats / np.linalg.norm(image_feats, ord=2, axis=-1, keepdims=True)
    text_feats = text_feats / np.linalg.norm(text_feats, ord=2, axis=-1, keepdims=True)
    # Get from dygraph， refer to predict.py
    exp_data = np.exp(args.temperature)
    m = softmax(np.matmul(exp_data * text_feats, image_feats.T), axis=0).T
    print(m)


if __name__ == "__main__":
    main()
