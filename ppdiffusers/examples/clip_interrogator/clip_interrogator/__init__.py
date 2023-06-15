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

from .clip_interrogator import Config, Interrogator

__version__ = "0.3.5"
__author__ = "pharmapsychotic"


CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # vit model
    "openai/clip-vit-base-patch32",  # ViT-B/32
    "openai/clip-vit-base-patch16",  # ViT-B/16
    "openai/clip-vit-large-patch14",  # ViT-L/14
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    # resnet model
    "openai/clip-rn50",  # RN50
    "openai/clip-rn101",  # RN101
    "openai/clip-rn50x4",  # RN50x4
]


BLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip-image-captioning-large",
]
