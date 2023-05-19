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

from clip_interrogator import (
    BLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
    CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
    Config,
    Interrogator,
)
from cog import BasePredictor, Input, Path
from PIL import Image


class Predictor(BasePredictor):
    def setup(self):
        self.ci = Interrogator(
            Config(
                blip_pretrained_model_name_or_path="Salesforce/blip-image-captioning-large",
                clip_pretrained_model_name_or_path="openai/clip-vit-large-patch14",
                device="gpu",
            )
        )

    def predict(
        self,
        image: Path = Input(description="Input image"),
        clip_pretrained_model_name_or_path: str = Input(
            default="openai/clip-vit-large-patch14",
            choices=CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            description="Choose ViT-L for Stable Diffusion 1, and ViT-H for Stable Diffusion 2",
        ),
        blip_pretrained_model_name_or_path: str = Input(
            default="Salesforce/blip-image-captioning-large",
            choices=BLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            description="Choose Salesforce/blip-image-captioning-large",
        ),
        mode: str = Input(
            default="best",
            choices=["best", "classic", "fast"],
            description="Prompt mode (best takes 10-20 seconds, fast takes 1-2 seconds).",
        ),
    ) -> str:
        """Run a single prediction on the model"""
        image = Image.open(str(image)).convert("RGB")
        self.switch_model(clip_pretrained_model_name_or_path, blip_pretrained_model_name_or_path)
        if mode == "best":
            return self.ci.interrogate(image)
        elif mode == "classic":
            return self.ci.interrogate_classic(image)
        else:
            return self.ci.interrogate_fast(image)

    def switch_model(self, clip_pretrained_model_name_or_path: str, blip_pretrained_model_name_or_path: str):
        if clip_pretrained_model_name_or_path != self.ci.config.clip_pretrained_model_name_or_path:
            self.ci.config.clip_pretrained_model_name_or_path = clip_pretrained_model_name_or_path
            self.ci.load_clip_model()
        if blip_pretrained_model_name_or_path != self.ci.config.blip_pretrained_model_name_or_path:
            self.ci.config.blip_pretrained_model_name_or_path = blip_pretrained_model_name_or_path
            self.ci.load_blip_model()
