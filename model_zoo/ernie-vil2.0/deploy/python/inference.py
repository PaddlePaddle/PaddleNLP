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

import argparse
import os

import numpy as np
import paddle
from PIL import Image

from paddlenlp.transformers import ErnieViLModel, ErnieViLProcessor
from paddlenlp.utils.downloader import get_path_from_url

MODEL_CLASSES = {
    "ernie_vil-2.0-base-zh": (ErnieViLModel, ErnieViLProcessor),
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
    def create_predictor(cls, args, model_path):
        config = paddle.inference.Config(model_path + ".pdmodel", model_path + ".pdiparams")
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
        input_handles = [predictor.get_input_handle(name) for name in predictor.get_input_names()]
        output_handles = [predictor.get_output_handle(name) for name in predictor.get_output_names()]
        return cls(predictor, input_handles, output_handles)

    def predict_batch(self, data):
        for input_field, input_handle in zip(data, self.input_handles):
            input_handle.copy_from_cpu(input_field.numpy() if isinstance(input_field, paddle.Tensor) else input_field)
        self.predictor.run()
        output = [output_handle.copy_to_cpu() for output_handle in self.output_handles]
        return output

    def predict(self, dataset, batch_size=1):
        outputs = []
        for data in dataset:
            output = self.predict_batch(data)
            outputs.append(output)
        return outputs


def get_test_image_path():
    image_path = "./tests/fixtures/tests_samples/COCO/000000039769.png"
    # Test image path for unittest
    if os.path.exists(image_path):
        return image_path
    else:
        # For normal test
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        get_path_from_url(url, root_dir=".")
        return "000000039769.jpg"


def main():
    args = parse_args()

    args.model_type = args.model_type.lower()
    model_class, processor_class = MODEL_CLASSES[args.model_type]
    processor = processor_class.from_pretrained("PaddlePaddle/ernie_vil-2.0-base-zh")
    text_model_path = os.path.join(args.model_path, "get_text_features")

    ds = [
        "这是一只猫",
        "这是一只狗",
    ]
    predictor = Predictor.create_predictor(args, text_model_path)
    dataset = [[np.array(processor(text)["input_ids"]).astype("int64").reshape([1, -1])] for text in ds]
    outs = predictor.predict(dataset)
    print(outs)
    image_path = get_test_image_path()
    image = Image.open(image_path)
    image_model_path = os.path.join(args.model_path, "get_image_features")
    predictor = Predictor.create_predictor(args, image_model_path)
    images = [[processor(images=image, return_tensors="np")["pixel_values"]]]
    outs = predictor.predict(images)
    print(outs)


if __name__ == "__main__":
    main()
