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

import paddle
import paddle.nn.functional as F
from PIL import Image

from paddlenlp.transformers import ErnieViLModel, ErnieViLProcessor, ErnieViLTokenizer

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--resume", default=None, type=str, help="path to latest checkpoint (default: none)",)
args = parser.parse_args()
# yapf: enable


def main():

    tokenizer = ErnieViLTokenizer.from_pretrained("ernie_vil-2.0-base-zh")
    if args.resume is not None:
        # Loading finetuned model
        model = ErnieViLModel.from_pretrained(args.resume)
        processor = ErnieViLProcessor.from_pretrained(
            "PaddlePaddle/ernie_vil-2.0-base-zh",
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
        )
        print("Loading parameters from " + args.resume)
    else:
        model = ErnieViLModel.from_pretrained("PaddlePaddle/ernie_vil-2.0-base-zh")
        processor = ErnieViLProcessor.from_pretrained("PaddlePaddle/ernie_vil-2.0-base-zh")

    model.eval()
    image = Image.open("examples/000000039769.jpeg")
    images = processor(images=image, return_tensors="pd")
    source_text = ["猫的照片", "狗的照片"]
    texts = tokenizer(source_text, padding=True, return_tensors="pd")
    with paddle.no_grad():
        image_features = model.get_image_features(**images)
        text_features = model.get_text_features(**texts)
        print("Image features:")
        print(image_features)
        print("Text features")
        print(text_features)
        # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
        image_features /= image_features.norm(axis=-1, keepdim=True)
        text_features /= text_features.norm(axis=-1, keepdim=True)
        # breakpoint()
        ret = model(pixel_values=images["pixel_values"], input_ids=texts["input_ids"])
        # logits_per_text = ret.logits_per_text
        logits_per_image = (ret.logits_per_image,)
        probs = F.softmax(logits_per_image, axis=-1)

    print("Label probs:", probs)


if __name__ == "__main__":
    main()
