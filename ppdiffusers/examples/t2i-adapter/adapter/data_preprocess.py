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


def vic_watermark(score_json, height, width):
    if "vic_watermark_info" in score_json:
        vic_watermark_score = score_json["vic_watermark_info"]
        if vic_watermark_score == 0:
            vic_watermark_info = 0
        elif len(vic_watermark_score) > 1:
            vic_watermark_info = 1
        else:
            left = vic_watermark_score[0]["left"]
            right = vic_watermark_score[0]["right"]
            bottom = vic_watermark_score[0]["bottom"]
            top = vic_watermark_score[0]["top"]
            # score = vic_watermark_score[0]["score"]
            area = ((bottom - top) * height) * ((right - left) * width) / (height * width)
            vic_watermark_info = area
    else:
        vic_watermark_info = 0
    return vic_watermark_info


def process_data(line, filename, data_format):
    try:
        data = line.strip().split("\t")
        # canny
        # if data_format == "img2img":
        #     text_id = data[0]
        #     text_json = json.loads(data[2])
        #     image_num = int(data[3]) # 2
        #     image_b64str = data[4 + image_num] # data[6]
        #     control_image_b64str = data[4 + image_num + 1] # data[7]
        #     resolution = data[4 + image_num + image_num] # data[8]
        #     control_resolution = data[4 + image_num + image_num + 1] # data[9]
        #     score_json = {} # TO ADD.
        # else:
        #     text_id = data[0]
        #     text_json = json.loads(data[2])
        #     image_b64str = data[5]
        #     score_json = json.loads(data[6])
        #     resolution = data[7]
        #     control_image_b64str = None

        # openpose
        data = line.strip().split("\t")
        if data_format == "img2img":
            text_id = data[0]
            text_json = json.loads(data[2])
            image_num = int(data[3])  # 2
            image_b64str = data[4 + image_num]  # data[6]
            control_image_b64str = data[4 + image_num + 1]  # data[7]
            if len(data) == 10:
                score_str = ""
                score_json = {}
                resolution = data[4 + image_num + image_num]  # data[8]
                # control_resolution = data[4 + image_num + image_num + 1]  # data[9]
            elif len(data) == 11:
                score_json_idx = 4 + image_num + image_num  # 8
                score_str = data[score_json_idx]
                score_json = json.loads(score_str)
                resolution = data[score_json_idx + 1]  # data[9]
                # control_resolution = data[score_json_idx + 2]  # data[10]
            else:
                assert False, "invalid img2img data"
        else:
            text_id = data[0]
            text_json = json.loads(data[2])
            image_b64str = data[5]
            score_str = data[6]
            score_json = json.loads(score_str)
            resolution = data[7]
            control_image_b64str = None
        if resolution != "X":
            try:
                width, height = float(resolution.split("X")[0].replace("X", "")), float(
                    resolution.split("X")[1].replace("X", "")
                )
            except:
                width, height = float(resolution.split("x")[0].replace("x", "")), float(
                    resolution.split("x")[1].replace("x", "")
                )
        else:
            tmp = base64_to_image(image_b64str)
            width, height = tmp.size[0], tmp.size[1]
        min_resolution = min(height, width)
        max_resolution = max(height, width)
        # vic_watermark_info = vic_watermark(score_json, height, width)

        # laion5B精选 laion-aes-v2
        if "laion_aes" in text_id:
            if (
                "AESTHETIC_SCORE" in score_json
                and score_json["AESTHETIC_SCORE"]
                and float(score_json["AESTHETIC_SCORE"]) < 6
            ):
                return None
            if min_resolution < 512 or max_resolution / min_resolution > 3:
                return None
            if "pwatermark" in score_json and score_json["pwatermark"] and float(score_json["pwatermark"]) > 0.3:
                return None
            if "ase_score" in score_json and float(score_json["ase_score"]) < 6:
                return None
            if "watermark_score" in score_json and score_json["watermark_score"] > 0.3:
                return None

            caption = ""
            # if "caption_zh" in text_json:
            #     # 如果翻译成功，则用翻译数据
            #     if text_json["caption_zh"] != text_json["caption_en"]:
            #         caption += text_json["caption_zh"].strip("\"")
            #     # 否则用图生文
            #     elif 'blip_caption_zh' in text_json:
            #         caption += text_json['blip_caption_zh']
            #     else:
            #         caption += text_json['caption_zh'].strip("\"")
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
