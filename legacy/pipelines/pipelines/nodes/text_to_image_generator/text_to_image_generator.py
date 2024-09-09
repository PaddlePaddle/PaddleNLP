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

import hashlib
import os
import time
from io import BytesIO
from typing import Optional

import requests
from PIL import Image
from pipelines.nodes.base import BaseComponent
from pipelines.schema import Document
from tqdm.auto import tqdm


class ErnieTextToImageGenerator(BaseComponent):
    """
    ErnieTextToImageGenerator that uses a Ernie Vilg for text to image generation.
    """

    def __init__(self, ak=None, sk=None):
        """
        :param ak: ak for applying token to request wenxin api.
        :param sk: sk for applying token to request wenxin api.
        """
        if ak is None or sk is None:
            raise Exception("Please apply api_key and secret_key from https://wenxin.baidu.com/moduleApi/ernieVilg")
        self.ak = ak
        self.sk = sk
        self.token_host = "https://wenxin.baidu.com/younger/portal/api/oauth/token"
        self.token = self._apply_token(self.ak, self.sk)

        # save init parameters to enable export of component config as YAML
        self.set_config(
            ak=ak,
            sk=sk,
        )

    def _apply_token(self, ak, sk):
        if ak is None or sk is None:
            ak = self.ak
            sk = self.sk
        response = requests.get(
            self.token_host, params={"grant_type": "client_credentials", "client_id": ak, "client_secret": sk}
        )
        if response:
            res = response.json()
            if res["code"] != 0:
                print("Request access token error.")
                raise RuntimeError("Request access token error.")
        else:
            print("Request access token error.")
            raise RuntimeError("Request access token error.")
        return res["data"]

    def generate_image(
        self,
        text_prompts,
        style: Optional[str] = "探索无限",
        resolution: Optional[str] = "1024*1024",
        topk: Optional[int] = 6,
        visualization: Optional[bool] = True,
        output_dir: Optional[str] = "ernievilg_output",
    ):
        """
        Create image by text prompts using ErnieVilG model.
        :param text_prompts: Phrase, sentence, or string of words and phrases describing what the image should look like.
        :param style: Image stype, currently supported 古风、油画、水彩、卡通、二次元、浮世绘、蒸汽波艺术、
        low poly、像素风格、概念艺术、未来主义、赛博朋克、写实风格、洛丽塔风格、巴洛克风格、超现实主义、探索无限。
        :param resolution: Resolution of images, currently supported "1024*1024", "1024*1536", "1536*1024".
        :param topk: Top k images to save.
        :param visualization: Whether to save images or not.
        :output_dir: Output directory
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        token = self.token
        create_url = "https://wenxin.baidu.com/younger/portal/api/rest/1.0/ernievilg/v1/txt2img?from=paddlehub"
        get_url = "https://wenxin.baidu.com/younger/portal/api/rest/1.0/ernievilg/v1/getImg?from=paddlehub"
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        taskids = []
        for text_prompt in text_prompts:
            res = requests.post(
                create_url,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={"access_token": token, "text": text_prompt, "style": style, "resolution": resolution},
            )
            res = res.json()
            if res["code"] == 4001:
                print("请求参数错误")
                raise RuntimeError("请求参数错误")
            elif res["code"] == 4002:
                print("请求参数格式错误，请检查必传参数是否齐全，参数类型等")
                raise RuntimeError("请求参数格式错误，请检查必传参数是否齐全，参数类型等")
            elif res["code"] == 4003:
                print("请求参数中，图片风格不在可选范围内")
                raise RuntimeError("请求参数中，图片风格不在可选范围内")
            elif res["code"] == 4004:
                print("API服务内部错误，可能引起原因有请求超时、模型推理错误等")
                raise RuntimeError("API服务内部错误，可能引起原因有请求超时、模型推理错误等")
            elif res["code"] == 100 or res["code"] == 110 or res["code"] == 111:
                token = self._apply_token(self.ak, self.sk)
                res = requests.post(
                    create_url,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    data={"access_token": token, "text": text_prompt, "style": style, "resolution": resolution},
                )
                res = res.json()
                if res["code"] != 0:
                    print("Token失效重新请求后依然发生错误，请检查输入的参数")
                    raise RuntimeError("Token失效重新请求后依然发生错误，请检查输入的参数")
            if res["msg"] == "success":
                taskids.append(res["data"]["taskId"])
            else:
                print(res["msg"])
                raise RuntimeError(res["msg"])

        start_time = time.time()
        process_bar = tqdm(total=100, unit="%")
        results = {}
        total_time = 60 * len(taskids)
        while True:
            end_time = time.time()
            duration = end_time - start_time
            progress_rate = int((duration) / total_time * 100)
            if not taskids:
                progress_rate = 100
            if progress_rate > process_bar.n:
                if progress_rate >= 100:
                    if not taskids:
                        increase_rate = 100 - process_bar.n
                    else:
                        increase_rate = 0
                else:
                    increase_rate = progress_rate - process_bar.n
            else:
                increase_rate = 0
            process_bar.update(increase_rate)
            if duration < 30:
                time.sleep(5)
                continue
            else:
                time.sleep(6)
            if not taskids:
                break
            has_done = []
            for taskid in taskids:
                res = requests.post(
                    get_url,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    data={"access_token": token, "taskId": {taskid}},
                )
                res = res.json()
                if res["code"] == 4001:
                    print("请求参数错误")
                    raise RuntimeError("请求参数错误")
                elif res["code"] == 4002:
                    print("请求参数格式错误，请检查必传参数是否齐全，参数类型等")
                    raise RuntimeError("请求参数格式错误，请检查必传参数是否齐全，参数类型等")
                elif res["code"] == 4003:
                    print("请求参数中，图片风格不在可选范围内")
                    raise RuntimeError("请求参数中，图片风格不在可选范围内")
                elif res["code"] == 4004:
                    print("API服务内部错误，可能引起原因有请求超时、模型推理错误等")
                    raise RuntimeError("API服务内部错误，可能引起原因有请求超时、模型推理错误等")
                elif res["code"] == 100 or res["code"] == 110 or res["code"] == 111:
                    token = self._apply_token(self.ak, self.sk)
                    res = requests.post(
                        get_url,
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                        data={"access_token": token, "taskId": {taskid}},
                    )
                    res = res.json()
                    if res["code"] != 0:
                        print("Token失效重新请求后依然发生错误，请检查输入的参数")
                        raise RuntimeError("Token失效重新请求后依然发生错误，请检查输入的参数")
                if res["msg"] == "success":
                    if res["data"]["status"] == 1:
                        has_done.append(res["data"]["taskId"])
                    results[res["data"]["text"]] = {
                        "imgUrls": res["data"]["imgUrls"],
                        "waiting": res["data"]["waiting"],
                        "taskId": res["data"]["taskId"],
                    }
                else:
                    print(res["msg"])
                    raise RuntimeError(res["msg"])
            for taskid in has_done:
                taskids.remove(taskid)
        print("Saving Images...")
        result_images = []
        for text, data in results.items():
            for idx, imgdata in enumerate(data["imgUrls"]):
                try:
                    image = Image.open(BytesIO(requests.get(imgdata["image"]).content))
                except Exception:
                    print("Download generated images error, retry one time")
                    try:
                        image = Image.open(BytesIO(requests.get(imgdata["image"]).content))
                    except Exception:
                        raise RuntimeError("Download generated images failed.")
                if visualization:
                    ext = "png"
                    md5hash = hashlib.md5(image.tobytes())
                    md5_name = md5hash.hexdigest()
                    image_name = "{}.{}".format(md5_name, ext)
                    image_path = os.path.join(output_dir, image_name)
                    image.save(image_path)
                    result_images.append(image_path)
                if idx + 1 >= topk:
                    break
        print("Done")
        return result_images

    def run(
        self,
        query: Document,
        style: Optional[str] = None,
        topk: Optional[int] = None,
        resolution: Optional[str] = "1024*1024",
        output_dir: Optional[str] = "ernievilg_output",
    ):

        result_images = self.generate_image(
            query, style=style, topk=topk, resolution=resolution, output_dir=output_dir
        )
        results = {"results": result_images}
        return results, "output_1"
