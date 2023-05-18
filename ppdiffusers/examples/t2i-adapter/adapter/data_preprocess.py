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

import base64
import io
import json
import random

from PIL import Image


def base64_to_image(base64_str):
    byte_data = base64.b64decode(base64_str)
    image_data = io.BytesIO(byte_data)
    img = Image.open(image_data)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def process_data(line, filename, data_format):
    try:
        data = line.strip().split("\t")
        if data_format == "img2img":
            text_id = data[0]
            text_json = json.loads(data[2])
            image_num = int(data[3])  # 2
            image_b64str = data[4 + image_num]  # data[6]
            control_image_b64str = data[4 + image_num + 1]  # data[7]
        else:
            text_id = data[0]
            text_json = json.loads(data[2])
            image_b64str = data[5]
            control_image_b64str = None

        caption = ""
        caption += text_json.get("caption_en", text_json.get("blip_caption_en", ""))
        if caption != "":
            image_base64 = image_b64str
        else:
            return None

        return image_base64, caption, text_id, control_image_b64str

    except Exception as e:
        print(f"error when parse file {filename}")
        print(e)
        return None


def parse_line(line, filename, data_format="default"):
    try:
        res = process_data(line, filename, data_format)
        if res is not None:
            image_base64, caption, _id, control_image_base64 = res
            image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
            if control_image_base64 is not None:
                image_extract = io.BytesIO(base64.b64decode(control_image_base64))
                control_image = Image.open(image_extract).convert("RGB")

                control_image = control_image.resize(image.size)
            else:
                control_image = None

            if image.size[0] < image.size[1]:  # 长图裁剪
                crop_size = (0, 0, image.size[0], image.size[0])
            else:  # 宽图裁剪
                crop_size = (
                    (image.size[0] - image.size[1]) // 2,
                    0,
                    (image.size[0] + image.size[1]) // 2,
                    image.size[1],
                )
            image = image.crop(crop_size)
            if control_image is not None:
                control_image = control_image.crop(crop_size)

            # drop out
            if random.random() < 0.5:
                caption = ""
            return dict(
                image=image,
                caption=caption,
                _id=_id,
                control_image=control_image,
            )
        else:
            return None
    except Exception as e:
        print(f"error when parse file {filename}")
        print(e)
        return None
