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

from .transform import (
    blip_transform,
    blip_transform_randaug,
    blip_transform_randaug_pretrain,
    blip_transform_randaug_wc,
    blip_transform_randaug_wohf,
    clip_transform,
    clip_transform_randaug,
    imagenet_transform,
    imagenet_transform_randaug,
    pixelbert_transform,
    pixelbert_transform_randaug,
    vit_transform,
    vit_transform_randaug,
)

_transforms = {
    "pixelbert": pixelbert_transform,
    "pixelbert_randaug": pixelbert_transform_randaug,
    "vit": vit_transform,
    "vit_randaug": vit_transform_randaug,
    "imagenet": imagenet_transform,
    "imagenet_randaug": imagenet_transform_randaug,
    "clip": clip_transform,
    "clip_randaug": clip_transform_randaug,
    "blip": blip_transform,
    "blip_randaug": blip_transform_randaug,
    "blip_randaug_wc": blip_transform_randaug_wc,
    "blip_randaug_wohf": blip_transform_randaug_wohf,
    "blip_randaug_pretrain": blip_transform_randaug_pretrain,
}


def keys_to_transforms(keys: list, size=224):
    return [_transforms[key](size=size) for key in keys]
