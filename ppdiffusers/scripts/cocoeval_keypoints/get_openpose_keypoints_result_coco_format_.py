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

import json
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import cv2
import numpy as np
import paddle
import paddlehub as hub

# import PIL
from PIL import Image

annotator_ckpts_path = os.path.join(os.path.dirname(__file__), "ckpts")


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("path", type=str, nargs=2, help=("Paths to the input dir and output dir"))

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}


class OpenposePaddleDetector:
    def __init__(self):
        self.body_estimation = hub.Module(name="openpose_body_estimation")

    def __call__(self, oriImg, hand=False):
        oriImg = oriImg[:, :, ::-1].copy()
        with paddle.no_grad():
            canvas = oriImg[:, :, ::-1].copy()
            canvas.fill(0)
            result = self.body_estimation.predict(oriImg, save_path="saved_images", visualization=False)

            canvas = self.body_estimation.draw_pose(canvas, result["candidate"], result["subset"])
        return canvas, dict(candidate=result["candidate"].tolist(), subset=result["subset"].tolist())


def get_keypoints_result_coco_format(paths, detector):
    """Get keypoints result in coco format"""
    if not os.path.exists(paths[0]):
        raise RuntimeError("Invalid path: %s" % paths[0])
    dir_path = pathlib.Path(paths[0])
    files = sorted([file for ext in IMAGE_EXTENSIONS for file in dir_path.glob("*.{}".format(ext))])
    output = []
    for file in files:
        im = Image.open(file)
        im = np.array(im, dtype=np.uint8)
        # breakpoint()

        input_image = HWC3(im)
        # detected_map, _ = detector(input_image)
        # detected_map = HWC3(detected_map)

        canvas, keypoints_result = detector(input_image)
        Image.fromarray(canvas).save("temp.png")
        sample_dict = {
            "image_id": 136,
            "category_id": 1,
            "keypoints": [
                36,
                181,
                2,
                20.25,
                191,
                0,
                35,
                166,
                2,
                20.25,
                191,
                0,
                8,
                171,
                2,
                20.25,
                191,
                0,
                2,
                246,
                2,
                20.25,
                191,
                0,
                20.25,
                191,
                0,
                20.25,
                191,
                0,
                20.25,
                191,
                0,
                20.25,
                191,
                0,
                20.25,
                191,
                0,
                20.25,
                191,
                0,
                20.25,
                191,
                0,
                20.25,
                191,
                0,
                20.25,
                191,
                0,
            ],
            "score": 0.897,
        }

        breakpoint()
        output.append(sample_dict)
    with open(paths[1], "w") as json_file:
        json_file.write(json.dumps(output, indent=4))


if __name__ == "__main__":
    args = parser.parse_args()
    detector = OpenposePaddleDetector()
    get_keypoints_result_coco_format(args.path, detector)
