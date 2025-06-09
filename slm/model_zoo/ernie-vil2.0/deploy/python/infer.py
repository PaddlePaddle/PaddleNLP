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
from PIL import Image
from scipy.special import softmax

from paddlenlp.transformers import ErnieViLProcessor
from paddlenlp.utils.env import (
    PADDLE_INFERENCE_MODEL_SUFFIX,
    PADDLE_INFERENCE_WEIGHTS_SUFFIX,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Directory with .json and .pdiparams")
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"], help="Device for inference")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=4.3)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--encode_type", choices=["text", "image"], default="text")
    parser.add_argument("--image_path", type=str, default="data/datasets/Flickr30k-CN/image/36979.jpg")
    return parser.parse_args()


class PaddleErnieViLPredictor:
    def __init__(self, args):
        self.args = args
        self.processor = ErnieViLProcessor.from_pretrained("PaddlePaddle/ernie_vil-2.0-base-zh")
        self.predictor, self.input_names, self.output_names = self.load_predictor()

    def load_predictor(self):
        model_file = os.path.join(
            self.args.model_dir, f"get_{self.args.encode_type}_features{PADDLE_INFERENCE_MODEL_SUFFIX}"
        )
        params_file = os.path.join(
            self.args.model_dir, f"get_{self.args.encode_type}_features{PADDLE_INFERENCE_WEIGHTS_SUFFIX}"
        )

        config = paddle_infer.Config(model_file, params_file)
        if self.args.device == "gpu":
            config.enable_use_gpu(100, 0)
        else:
            config.disable_gpu()
        config.disable_glog_info()
        config.switch_ir_optim(True)

        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()
        output_names = predictor.get_output_names()
        return predictor, input_names, output_names

    def preprocess(self, inputs):
        if self.args.encode_type == "text":
            input_ids = [self.processor(text=t)["input_ids"] for t in inputs]
            input_ids = np.array(input_ids, dtype="int64")
            return {"input_ids": input_ids}
        else:
            pixel_values = [self.processor(images=img)["pixel_values"][0] for img in inputs]
            pixel_values = np.stack(pixel_values)
            return {"pixel_values": pixel_values.astype("float32")}

    def infer(self, input_dict):
        for name in self.input_names:
            input_tensor = self.predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(input_dict[name])
        self.predictor.run()
        output_tensor = self.predictor.get_output_handle(self.output_names[0])
        return output_tensor.copy_to_cpu()

    def predict(self, inputs):
        input_map = self.preprocess(inputs)
        output = self.infer(input_map)
        return output


def main():
    args = parse_arguments()

    # 文本推理
    args.encode_type = "text"
    predictor_text = PaddleErnieViLPredictor(args)
    texts = ["猫的照片", "狗的照片"]
    args.batch_size = len(texts)
    text_features = predictor_text.predict(texts)

    # 图像推理
    args.encode_type = "image"
    args.batch_size = 1
    predictor_image = PaddleErnieViLPredictor(args)
    image = Image.open(args.image_path).convert("RGB")
    image_features = predictor_image.predict([image])

    # 特征归一化 + 相似度计算
    image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)
    text_features = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)

    sim_logits = softmax(np.exp(args.temperature) * np.matmul(text_features, image_features.T), axis=0).T
    print("相似度矩阵（image→text）:")
    print(sim_logits)


if __name__ == "__main__":
    main()
