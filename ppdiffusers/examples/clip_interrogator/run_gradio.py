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

import gradio as gr
import paddle
from clip_interrogator import (
    BLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
    CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
    Config,
    Interrogator,
)

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
parser.add_argument("-s", "--share", action="store_true", help="Create a public link")
parser.add_argument("--server_name", default="0.0.0.0", type=str, help="server_name")
parser.add_argument("--server_port", default=8586, type=int, help="server_port")

args = parser.parse_args()

# validate clip model name
if args.clip not in CLIP_PRETRAINED_MODEL_ARCHIVE_LIST:
    clip_models = ", ".join(CLIP_PRETRAINED_MODEL_ARCHIVE_LIST)
    print(f"Could not find CLIP model {args.clip}!")
    print(f"    available clip models: {clip_models}")
    exit(1)

# validate clip model name
if args.blip not in BLIP_PRETRAINED_MODEL_ARCHIVE_LIST:
    blip_models = ", ".join(BLIP_PRETRAINED_MODEL_ARCHIVE_LIST)
    print(f"Could not find BLIP model {args.blip}!")
    print(f"    available blip models: {blip_models}")
    exit(1)

# select device
if args.device == "auto":
    device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
else:
    device = args.device
paddle.set_device(device)
config = Config(
    cache_path="cache", clip_pretrained_model_name_or_path=args.clip, blip_pretrained_model_name_or_path=args.blip
)
ci = Interrogator(config)


def inference(
    image,
    mode,
    clip_pretrained_model_name_or_path,
    blip_pretrained_model_name_or_path,
    blip_min_length,
    blip_max_length,
    blip_sample,
    blip_top_p,
    blip_repetition_penalty,
    blip_num_beams,
):
    if clip_pretrained_model_name_or_path != ci.config.clip_pretrained_model_name_or_path:
        ci.config.clip_pretrained_model_name_or_path = clip_pretrained_model_name_or_path
        ci.load_clip_model()

    if blip_pretrained_model_name_or_path != ci.config.blip_pretrained_model_name_or_path:
        ci.config.blip_pretrained_model_name_or_path = blip_pretrained_model_name_or_path
        ci.load_blip_model()

    ci.config.blip_min_length = int(blip_min_length)
    ci.config.blip_max_length = int(blip_max_length)
    ci.config.blip_sample = eval(blip_sample)
    ci.config.blip_top_p = float(blip_top_p)
    ci.config.blip_repetition_penalty = float(blip_repetition_penalty)
    ci.config.blip_num_beams = int(blip_num_beams)

    image = image.convert("RGB")
    if mode == "best":
        return ci.interrogate(image)
    elif mode == "classic":
        return ci.interrogate_classic(image)
    else:
        return ci.interrogate_fast(image)


inputs = [
    gr.inputs.Image(type="pil"),
    gr.Radio(["best", "classic", "fast"], label="Mode", value="fast"),
    gr.Dropdown(CLIP_PRETRAINED_MODEL_ARCHIVE_LIST, value=args.clip, label="CLIP Model"),
    gr.Dropdown(BLIP_PRETRAINED_MODEL_ARCHIVE_LIST, value=args.blip, label="BLIP Model"),
    gr.Number(value=8, label="Caption min Length"),
    gr.Number(value=32, label="Caption Max Length"),
    gr.Radio(["True", "False"], value="False", label="Sample or not?"),
    gr.Number(value=0.9, label="TopP value, when Sample is true"),
    gr.Number(value=1.1, label="Repetition penalty value, when Sample is false"),
    gr.Number(value=64, label="Caption Num Beams, when Sample is false"),
]
outputs = [
    gr.outputs.Textbox(label="Image Caption Output"),
]

io = gr.Interface(
    inference,
    inputs,
    outputs,
    title="🕵️‍♂️ Paddle CLIP Interrogator 🕵️‍♂️",
    allow_flagging=False,
)
io.launch(share=args.share, server_name=args.server_name, server_port=args.server_port)
