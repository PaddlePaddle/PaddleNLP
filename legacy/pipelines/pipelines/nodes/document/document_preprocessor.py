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

import base64
import logging
import os
from io import BytesIO

import numpy as np
import paddle
from paddleocr import PaddleOCR
from PIL import Image
from pipelines.nodes.base import BaseComponent

from paddlenlp.taskflow.utils import download_file

logger = logging.getLogger(__name__)


class DocOCRProcessor(BaseComponent):
    """
    Preprocess document input from image/image url/image bytestream to ocr outputs
    """

    return_no_answers: bool
    outgoing_edges = 1
    query_count = 0
    query_time = 0

    def __init__(self, use_gpu: bool = True, lang: str = "ch"):
        """
        Init Document Preprocessor.
        :param use_gpu: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
        :param lang: Choose ocr model processing langugae
        """
        self._lang = lang
        self._use_gpu = False if paddle.get_device() == "cpu" else use_gpu
        self._ocr = PaddleOCR(use_angle_cls=True, show_log=False, use_gpu=self._use_gpu, lang=self._lang)

    def _check_input_text(self, inputs):
        if isinstance(inputs, dict):
            inputs = [inputs]
        if isinstance(inputs, list):
            input_list = []
            for example in inputs:
                data = {}
                if isinstance(example, dict):
                    if "doc" not in example.keys():
                        raise ValueError(
                            "Invalid inputs, the inputs should contain an url to an image or a local path."
                        )
                    else:
                        if isinstance(example["doc"], str):

                            if example["doc"].startswith("http://") or example["doc"].startswith("https://"):
                                download_file("./", example["doc"].rsplit("/", 1)[-1], example["doc"])
                                data["doc"] = example["doc"].rsplit("/", 1)[-1]
                            elif os.path.isfile(example["doc"]):
                                data["doc"] = example["doc"]
                            else:
                                img = base64.b64decode(example["doc"].encode("utf-8"))
                                img = np.frombuffer(bytearray(img), dtype="uint8")
                                img = np.array(Image.open(BytesIO(img)).convert("RGB"))
                                img = Image.fromarray(img)
                                img.save("./tmp.jpg")
                                data["doc"] = "./tmp.jpg"
                        else:
                            raise ValueError("Incorrect path or url, URLs must start with `http://` or `https://`")
                    if "prompt" not in example.keys():
                        raise ValueError("Invalid inputs, the inputs should contain the prompt.")
                    else:
                        if isinstance(example["prompt"], str):
                            data["prompt"] = [example["prompt"]]
                        elif isinstance(example["prompt"], list) and all(
                            isinstance(s, str) for s in example["prompt"]
                        ):
                            data["prompt"] = example["prompt"]
                        else:
                            raise TypeError("Incorrect prompt, prompt should be string or list of string.")
                    if "word_boxes" in example.keys():
                        data["word_boxes"] = example["word_boxes"]
                    input_list.append(data)
                else:
                    raise TypeError(
                        "Invalid inputs, input for document intelligence task should be dict or list of dict, but type of {} found!".format(
                            type(example)
                        )
                    )
        else:
            raise TypeError(
                "Invalid inputs, input for document intelligence task should be dict or list of dict, but type of {} found!".format(
                    type(inputs)
                )
            )
        return input_list

    def run(self, meta: dict):
        example = self._check_input_text(meta)[0]

        if "word_boxes" in example.keys():
            ocr_result = example["word_boxes"]
            example["ocr_type"] = "word_boxes"
        else:
            ocr_result = self._ocr.ocr(example["doc"], cls=True)
            example["ocr_type"] = "ppocr"
            # Compatible with paddleocr>=2.6.0.2
            ocr_result = ocr_result[0] if len(ocr_result) == 1 else ocr_result
        example["ocr_result"] = ocr_result
        output = {"example": example}
        return output, "output_1"
