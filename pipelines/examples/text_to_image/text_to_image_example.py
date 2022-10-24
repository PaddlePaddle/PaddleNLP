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

import os
import argparse

import paddle
from pipelines.nodes import ErnieTextToImageGenerator
from pipelines import TextToImagePipeline

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--api_key", default=None, type=str, help="The API Key.")
parser.add_argument("--secret_key", default=None, type=str, help="The secret key.")
parser.add_argument("--prompt_text", default='宁静的小镇', type=str, help="The prompt_text.")
parser.add_argument("--output_dir", default='ernievilg_output', type=str, help="The output path.")
parser.add_argument("--style", default='探索无限', type=str, help="The style text.")
parser.add_argument("--size", default='1024*1024',
    choices=['1024*1024', '1024*1536', '1536*1024'], help="Size of the generation images")
parser.add_argument("--topk", default=5, type=int, help="The top k images.")
args = parser.parse_args()
# yapf: enable


def text_to_image():
    erine_image_generator = ErnieTextToImageGenerator(ak=args.api_key,
                                                      sk=args.secret_key)
    pipe = TextToImagePipeline(erine_image_generator)
    prediction = pipe.run(query=args.prompt_text,
                          params={
                              "TextToImageGenerator": {
                                  "topk": args.topk,
                                  "style": args.style,
                                  "resolution": args.size,
                                  "output_dir": args.output_dir
                              }
                          })
    pipe.save_to_yaml('text_to_image.yaml')


if __name__ == "__main__":
    text_to_image()
