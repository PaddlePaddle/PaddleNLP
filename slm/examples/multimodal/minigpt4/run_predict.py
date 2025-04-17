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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["FLAGS_use_cuda_managed_memory"] = "true"
import requests
from PIL import Image

from paddlenlp.transformers import MiniGPT4ForConditionalGeneration, MiniGPT4Processor


def predict(args):
    # load MiniGPT4 moel and processor
    model = MiniGPT4ForConditionalGeneration.from_pretrained(args.pretrained_name_or_path)
    model.eval()
    processor = MiniGPT4Processor.from_pretrained(args.pretrained_name_or_path)
    print("load processor and model done!")

    # prepare model inputs for MiniGPT4
    url = "https://paddlenlp.bj.bcebos.com/data/images/mugs.png"
    image = Image.open(requests.get(url, stream=True).raw)

    text = "describe this image"
    prompt = "Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.###Human: <Img><ImageHere></Img> <TextHere>###Assistant:"
    inputs = processor([image], text, prompt)

    # generate with MiniGPT4
    # breakpoint
    generate_kwargs = {
        "max_length": 300,
        "num_beams": 1,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "length_penalty": 0,
        "temperature": 1,
        "decode_strategy": "greedy_search",
        "eos_token_id": [[835], [2277, 29937]],
    }
    outputs = model.generate(**inputs, **generate_kwargs)
    msg = processor.batch_decode(outputs[0])
    print("Inference result: ", msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_name_or_path",
        default="your directory of minigpt4",
        type=str,
        help="The dir name of minigpt4 checkpoint.",
    )
    args = parser.parse_args()

    predict(args)
