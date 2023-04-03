#!/usr/bin/env python3

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import csv
import os

import paddle
import requests
from clip_interrogator import (
    BLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
    CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
    Config,
    Interrogator,
)
from PIL import Image


def inference(ci, image, mode):
    image = image.convert("RGB")
    if mode == "best":
        return ci.interrogate(image)
    elif mode == "classic":
        return ci.interrogate_classic(image)
    else:
        return ci.interrogate_fast(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--clip",
        default="openai/clip-vit-large-patch14",
        choices=CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
        help="name of CLIP model to use",
    )
    parser.add_argument(
        "-b",
        "--blip",
        default="Salesforce/blip-image-captioning-large",
        choices=BLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
        help="name of BLIP model to use",
    )
    parser.add_argument("-d", "--device", default="auto", help="device to use (auto, gpu or cpu)")
    parser.add_argument("-f", "--folder", help="path to folder of images")
    parser.add_argument("-i", "--image", help="image file or url")
    parser.add_argument(
        "-m", "--mode", default="best", choices=["best", "classic", "fast"], help="best, classic, or fast"
    )

    args = parser.parse_args()
    if not args.folder and not args.image:
        parser.print_help()
        exit(1)

    if args.folder is not None and args.image is not None:
        print("Specify a folder or batch processing or a single image, not both")
        exit(1)

    # validate clip model name
    if args.clip not in CLIP_PRETRAINED_MODEL_ARCHIVE_LIST:
        models = ", ".join(CLIP_PRETRAINED_MODEL_ARCHIVE_LIST)
        print(f"Could not find CLIP model {args.clip}!")
        print(f"    available models: {models}")
        exit(1)

    # validate clip model name
    if args.blip not in BLIP_PRETRAINED_MODEL_ARCHIVE_LIST:
        models = ", ".join(BLIP_PRETRAINED_MODEL_ARCHIVE_LIST)
        print(f"Could not find BLIP model {args.blip}!")
        print(f"    available models: {models}")
        exit(1)

    # select device
    if args.device == "auto":
        device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
    else:
        device = args.device
    paddle.set_device(device)
    # generate a nice prompt
    config = Config(clip_pretrained_model_name_or_path=args.clip, blip_pretrained_model_name_or_path=args.blip)
    ci = Interrogator(config)

    # process single image
    if args.image is not None:
        image_path = args.image
        if str(image_path).startswith("http://") or str(image_path).startswith("https://"):
            image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        if not image:
            print(f"Error opening image {image_path}")
            exit(1)
        print(inference(ci, image, args.mode))

    # process folder of images
    elif args.folder is not None:
        if not os.path.exists(args.folder):
            print(f"The folder {args.folder} does not exist!")
            exit(1)

        files = [f for f in os.listdir(args.folder) if f.endswith(".jpg") or f.endswith(".png")]
        prompts = []
        for file in files:
            image = Image.open(os.path.join(args.folder, file)).convert("RGB")
            prompt = inference(ci, image, args.mode)
            prompts.append(prompt)
            print(prompt)

        if len(prompts):
            csv_path = os.path.join(args.folder, "desc.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                w.writerow(["image", "prompt"])
                for file, prompt in zip(files, prompts):
                    w.writerow([file, prompt])

            print(f"\n\n\n\nGenerated {len(prompts)} and saved to {csv_path}, enjoy!")


if __name__ == "__main__":
    main()
