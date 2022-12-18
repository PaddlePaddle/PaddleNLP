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

from paddle.vision import transforms
from paddle.vision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from PIL import Image

# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip
from .randaug import RandAugment
from .randaugment import RandomAugment
from .utils import MinMaxResize, imagenet_normalize, inception_normalize


def pixelbert_transform(size=800):
    longer = int((1333 / 800) * size)
    return transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )


def pixelbert_transform_randaug(size=800):
    longer = int((1333 / 800) * size)
    trs = transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))
    return trs


def imagenet_transform(size=800):
    return transforms.Compose(
        [
            Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            CenterCrop(size),
            transforms.ToTensor(),
            imagenet_normalize,
        ]
    )


def imagenet_transform_randaug(size=800):
    trs = transforms.Compose(
        [
            Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            CenterCrop(size),
            transforms.ToTensor(),
            imagenet_normalize,
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))
    return trs


def vit_transform(size=800):
    return transforms.Compose(
        [
            Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            CenterCrop(size),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )


def vit_transform_randaug(size=800):
    trs = transforms.Compose(
        [
            Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            CenterCrop(size),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))
    return trs


def clip_transform(size):
    return Compose(
        [
            Resize(size, interpolation="bicubic"),
            CenterCrop(size),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )


# def clip_transform(size):
#     return Compose([
#         Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
#         CenterCrop(size),
#         lambda image: image.convert("RGB"),
#         ToTensor(),
#         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#     ])


def clip_transform_randaug(size):
    trs = Compose(
        [
            Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            CenterCrop(size),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )
    trs.transforms.insert(0, lambda image: image.convert("RGBA"))
    trs.transforms.insert(0, RandAugment(2, 9))
    trs.transforms.insert(0, lambda image: image.convert("RGB"))
    return trs


def blip_transform(size):
    return Compose(
        [
            Resize((size, size), interpolation="bicubic"),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )


def blip_transform_randaug_pretrain(size):
    return Compose(
        [
            RandomResizedCrop(size, scale=(0.2, 1.0), interpolation="bicubic"),
            RandomHorizontalFlip(),
            RandomAugment(
                2,
                5,
                isPIL=True,
                augs=[
                    "Identity",
                    "AutoContrast",
                    "Brightness",
                    "Sharpness",
                    "Equalize",
                    "ShearX",
                    "ShearY",
                    "TranslateX",
                    "TranslateY",
                    "Rotate",
                ],
            ),
            lambda image: Image.fromarray(image).convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )


def blip_transform_randaug(size):
    return Compose(
        [
            RandomResizedCrop(size, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),
            RandomAugment(
                2,
                5,
                isPIL=True,
                augs=[
                    "Identity",
                    "AutoContrast",
                    "Brightness",
                    "Sharpness",
                    "Equalize",
                    "ShearX",
                    "ShearY",
                    "TranslateX",
                    "TranslateY",
                    "Rotate",
                ],
            ),
            lambda image: Image.fromarray(image).convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )


def blip_transform_randaug_wc(size):  # wc: with color changes
    return Compose(
        [
            RandomResizedCrop(size, scale=(0.5, 1.0), interpolation="bicubic"),
            RandomHorizontalFlip(),
            RandAugment(2, 7),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )


# def blip_transform_randaug_wohf(size): # remove horizontal flip for vqa
#     return Compose([
#             RandomResizedCrop(size, scale=(0.5, 1.0),interpolation=transforms.InterpolationMode.BICUBIC),
#             # RandomHorizontalFlip(),
#             RandomAugment(2, 5, isPIL=True, augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
#                                               'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
#             lambda image: Image.fromarray(image).convert("RGB"),
#             ToTensor(),
#             Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#         ])


def blip_transform_randaug_wohf(size):  # remove horizontal flip for vqa
    return Compose(
        [
            RandomResizedCrop(size, scale=(0.5, 1.0), interpolation="bicubic"),
            # RandomHorizontalFlip(),
            RandomAugment(
                2,
                5,
                isPIL=True,
                augs=[
                    "Identity",
                    "AutoContrast",
                    "Brightness",
                    "Sharpness",
                    "Equalize",
                    "ShearX",
                    "ShearY",
                    "TranslateX",
                    "TranslateY",
                    "Rotate",
                ],
            ),
            lambda image: Image.fromarray(image).convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )
