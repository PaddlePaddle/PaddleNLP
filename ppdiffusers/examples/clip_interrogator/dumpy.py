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

import gradio as gr
from clip_interrogator import (
    BLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
    CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
    Config,
    Interrogator,
)

blip_pretrained_model_name_or_path = "Salesforce/blip-image-captioning-base"
clip_pretrained_model_name_or_path = "openai/clip-vit-large-patch14"

# validate clip model name
if clip_pretrained_model_name_or_path not in CLIP_PRETRAINED_MODEL_ARCHIVE_LIST:
    clip_models = ", ".join(CLIP_PRETRAINED_MODEL_ARCHIVE_LIST)
    print(f"Could not find CLIP model {clip_pretrained_model_name_or_path}!")
    print(f"    available clip models: {clip_models}")
    exit(1)

# validate clip model name
if blip_pretrained_model_name_or_path not in BLIP_PRETRAINED_MODEL_ARCHIVE_LIST:
    blip_models = ", ".join(BLIP_PRETRAINED_MODEL_ARCHIVE_LIST)
    print(f"Could not find BLIP model {blip_pretrained_model_name_or_path}!")
    print(f"    available blip models: {blip_models}")
    exit(1)

config = Config(
    blip_num_beams=64,
    blip_pretrained_model_name_or_path=blip_pretrained_model_name_or_path,
    clip_pretrained_model_name_or_path=clip_pretrained_model_name_or_path,
)
ci = Interrogator(config)


def inference(image, mode, best_max_flavors=32):
    ci.config.chunk_size = (
        2048 if ci.config.clip_pretrained_model_name_or_path == "openai/clip-vit-large-patch14" else 1024
    )
    ci.config.flavor_intermediate_count = (
        2048 if ci.config.clip_pretrained_model_name_or_path == "openai/clip-vit-large-patch14" else 1024
    )
    image = image.convert("RGB")
    if mode == "best":
        return ci.interrogate(image, max_flavors=int(best_max_flavors))
    elif mode == "classic":
        return ci.interrogate_classic(image)
    else:
        return ci.interrogate_fast(image)


inputs = [
    gr.inputs.Image(type="pil"),
    gr.Radio(["best", "fast", "classic"], label="", value="best"),
    gr.Number(value=16, label="best mode max flavors"),
]
outputs = [
    gr.outputs.Textbox(label="Output"),
]

io = gr.Interface(
    inference,
    inputs,
    outputs,
    allow_flagging=False,
)
io.launch(debug=False, server_name="0.0.0.0", server_port=8586)
