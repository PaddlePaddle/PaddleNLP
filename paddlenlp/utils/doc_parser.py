# coding=utf-8
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
import mimetypes
import os
import random
import re
from io import BytesIO

import numpy as np
import requests
from PIL import Image, ImageDraw

from .image_utils import np2base64
from .log import logger


class DocParser(object):
    """DocParser"""

    def __init__(self, ocr_lang="ch", layout_analysis=False, pdf_parser_config=None, use_gpu=None, device_id=None):
        self.ocr_lang = ocr_lang
        self.use_angle_cls = False
        self.layout_analysis = layout_analysis
        self.pdf_parser_config = pdf_parser_config
        self.ocr_infer_model = None
        self.use_gpu = use_gpu
        self.device_id = device_id

    def parse(self, doc, expand_to_a4_size=False, do_ocr=True):
        """
        parse
        """
        doc_type = mimetypes.guess_type(doc["doc"])[0]

        if not doc_type or doc_type.startswith("image"):
            image = self.read_image(doc["doc"])
        elif doc_type == "application/pdf":
            image = self.read_pdf(doc["doc"])
        offset_x, offset_y = 0, 0
        if expand_to_a4_size:
            image, offset_x, offset_y = self.expand_image_to_a4_size(image, center=True)
        img_h, img_w = image.shape[:2]
        doc["image"] = np2base64(image)
        doc["offset_x"] = offset_x
        doc["offset_y"] = offset_y
        doc["img_w"] = img_w
        doc["img_h"] = img_h
        if do_ocr:
            ocr_result = self.ocr(image)
            if expand_to_a4_size:
                layout = []
                for segment in ocr_result:
                    box = segment[0]
                    org_box = [box[0] - offset_x, box[1] - offset_y, box[2] - offset_x, box[3] - offset_y]
                    layout.append((org_box, segment[1]))
                doc["layout"] = layout
            else:
                doc["layout"] = ocr_result
        return doc

    def __call__(self, *args, **kwargs):
        """
        Call parse
        """
        return self.parse(*args, **kwargs)

    def ocr(self, image, det=True, rec=True, cls=None):
        """
        Call ocr for an image
        """

        def _get_box(box):
            box = [
                min(box[0][0], box[3][0]),  # x1
                min(box[0][1], box[1][1]),  # y1
                max(box[1][0], box[2][0]),  # x2
                max(box[2][1], box[3][1]),  # y2
            ]
            return box

        def _is_ch(s):
            for ch in s:
                if "\u4e00" <= ch <= "\u9fff":
                    return True
            return False

        if self.ocr_infer_model is None:
            self.init_ocr_inference()
        if cls is None:
            cls = self.use_angle_cls

        layout = []
        if not self.layout_analysis:
            ocr_result = self.ocr_infer_model.ocr(image, det, rec, cls)
            ocr_result = ocr_result[0] if len(ocr_result) == 1 else ocr_result
            for segment in ocr_result:
                box = segment[0]
                box = _get_box(box)
                text = segment[1][0]
                layout.append((box, text))
        else:
            layout_result = self.layout_analysis_engine(image)
            for region in layout_result:
                if region["type"] != "table":
                    ocr_result = region["res"]
                    for segment in ocr_result:
                        box = segment["text_region"]
                        box = _get_box(box)
                        text = segment["text"]
                        layout.append((box, text))
                else:
                    bbox = region["bbox"]
                    table_result = region["res"]
                    html = table_result["html"]
                    cell_bbox = table_result["cell_bbox"]
                    table_list = []
                    lines = re.findall("<tr>(.*?)</tr>", html)
                    for line in lines:
                        table_list.extend(re.findall("<td.*?>(.*?)</td>", line))
                    for cell_box, text in zip(cell_bbox, table_list):
                        box = [
                            bbox[0] + cell_box[0],
                            bbox[3] - cell_box[5],
                            bbox[0] + cell_box[4],
                            bbox[3] - cell_box[1],
                        ]
                        if _is_ch(text):
                            text = text.replace(" ", "")
                        layout.append((box, text))
        return layout

    @classmethod
    def _get_buffer(self, data, file_like=False):
        buff = None
        if len(data) < 1024:
            if os.path.exists(data):
                buff = open(data, "rb").read()
            elif data.startswith("http://") or data.startswith("https://"):
                resp = requests.get(data, stream=True)
                if not resp.ok:
                    raise RuntimeError("Failed to download the file from {}".format(data))
                buff = resp.raw.read()
            else:
                raise FileNotFoundError("Image file {} not found!".format(data))
        if buff is None:
            buff = base64.b64decode(data)
        if buff and file_like:
            return BytesIO(buff)
        return buff

    @classmethod
    def read_image(self, image):
        """
        read image to np.ndarray
        """
        image_buff = self._get_buffer(image)

        _image = np.array(Image.open(BytesIO(image_buff)).convert("RGB"))
        return _image

    @classmethod
    def read_pdf(self, pdf, password=None):
        """
        read pdf
        """
        try:
            import fitz
        except ImportError:
            raise RuntimeError(
                "Need PyMuPDF to process pdf input. " "Please install module by: python3 -m pip install pymupdf"
            )
        if isinstance(pdf, fitz.Document):
            return pdf
        pdf_buff = self._get_buffer(pdf)
        if not pdf_buff:
            logger.warning("Failed to read pdf: %s...", pdf[:32])
            return None
        pdf_doc = fitz.Document(stream=pdf_buff, filetype="pdf")
        if pdf_doc.needs_pass:
            if pdf_doc.authenticate(password) == 0:
                raise ValueError("The password of pdf is incorrect.")

        if pdf_doc.page_count > 1:
            logger.warning("Currently only parse the first page for PDF input with more than one page.")

        page = pdf_doc.load_page(0)
        image = np.array(self.get_page_image(page).convert("RGB"))
        return image

    @classmethod
    def get_page_image(self, page):
        """
        get page image
        """
        pix = page.get_pixmap()
        image_buff = pix.pil_tobytes("jpeg", optimize=True)
        return Image.open(BytesIO(image_buff))

    def init_ocr_inference(self):
        """
        init ocr inference
        """
        if self.ocr_infer_model is not None:
            logger.warning("ocr model has already been initialized")
            return

        if not self.layout_analysis:
            try:
                from paddleocr import PaddleOCR
            except ImportError:
                raise RuntimeError(
                    "Need paddleocr to process image input. "
                    "Please install module by: python3 -m pip install paddleocr"
                )
            self.ocr_infer_model = PaddleOCR(show_log=False, lang=self.ocr_lang)
        else:
            try:
                from paddleocr import PPStructure
            except ImportError:
                raise RuntimeError(
                    "Need paddleocr to process image input. "
                    "Please install module by: python3 -m pip install paddleocr"
                )
            self.layout_analysis_engine = PPStructure(table=True, ocr=True, show_log=False, lang=self.ocr_lang)

    @classmethod
    def _normalize_box(self, box, old_size, new_size, offset_x=0, offset_y=0):
        """normalize box"""
        return [
            int((box[0] + offset_x) * new_size[0] / old_size[0]),
            int((box[1] + offset_y) * new_size[1] / old_size[1]),
            int((box[2] + offset_x) * new_size[0] / old_size[0]),
            int((box[3] + offset_y) * new_size[1] / old_size[1]),
        ]

    @classmethod
    def expand_image_to_a4_size(self, image, center=False):
        """expand image to a4 size"""
        h, w = image.shape[:2]
        offset_x, offset_y = 0, 0
        if h * 1.0 / w >= 1.42:
            exp_w = int(h / 1.414 - w)
            if center:
                offset_x = int(exp_w / 2)
                exp_img = np.zeros((h, offset_x, 3), dtype="uint8")
                exp_img.fill(255)
                image = np.hstack([exp_img, image, exp_img])
            else:
                exp_img = np.zeros((h, exp_w, 3), dtype="uint8")
                exp_img.fill(255)
                image = np.hstack([image, exp_img])
        elif h * 1.0 / w <= 1.40:
            exp_h = int(w * 1.414 - h)
            if center:
                offset_y = int(exp_h / 2)
                exp_img = np.zeros((offset_y, w, 3), dtype="uint8")
                exp_img.fill(255)
                image = np.vstack([exp_img, image, exp_img])
            else:
                exp_img = np.zeros((exp_h, w, 3), dtype="uint8")
                exp_img.fill(255)
                image = np.vstack([image, exp_img])
        return image, offset_x, offset_y

    @classmethod
    def write_image_with_results(
        self, image, layouts=None, results=None, save_path=None, return_pil_image=False, format=None, max_size=None
    ):
        """
        write image with boxes and results
        """

        def _flatten_results(results):
            """flatten results"""
            is_single = False
            if not isinstance(results, list):
                results = [results]
                is_single = True
            flat_results = []

            def _flatten(result):
                flat_result = []
                for key, vals in result.items():
                    for val in vals:
                        new_val = val.copy()
                        if val.get("relations"):
                            new_val["relations"] = _flatten(val["relations"])
                        new_val["label"] = key
                        flat_result.append(new_val)
                return flat_result

            for result in results:
                flat_results.append(_flatten(result))
            if is_single:
                return flat_results[0]
            return flat_results

        def _write_results(results):
            for segment in results:
                if isinstance(segment, dict):
                    boxes = segment["bbox"]
                else:
                    boxes = segment[0]
                if not isinstance(boxes[0], list):
                    boxes = [boxes]
                boxes = [
                    [
                        (box[0], box[1]),
                        (box[2], box[1]),
                        (box[2], box[3]),
                        (box[0], box[3]),
                    ]
                    for box in boxes
                ]
                while True:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    if sum(color) < 480:
                        break
                for box in boxes:
                    draw_render.polygon(box, fill=color)
                if isinstance(segment, dict) and segment.get("relations"):
                    _write_results(segment["relations"])

        random.seed(0)
        _image = self.read_image(image)
        _image = Image.fromarray(np.uint8(_image))
        h, w = _image.height, _image.width
        img_render = _image.copy()
        draw_render = ImageDraw.Draw(img_render)

        if layouts:
            for segment in layouts:
                if isinstance(segment, dict):
                    box = segment["bbox"]
                else:
                    box = segment[0]
                box = [
                    (box[0], box[1]),
                    (box[2], box[1]),
                    (box[2], box[3]),
                    (box[0], box[3]),
                ]
                while True:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    if sum(color) < 480:
                        break
                draw_render.polygon(box, fill=color)

        elif results:
            results = _flatten_results(results)
            _write_results(results)

        img_render = Image.blend(_image, img_render, 0.3)
        img_show = Image.new("RGB", (w, h), (255, 255, 255))
        img_show.paste(img_render, (0, 0, w, h))
        w, h = img_show.width, img_show.height
        if max_size and max(w, h) > max_size:
            if max(w, h) == h:
                new_size = (int(w * max_size / h), max_size)
            else:
                new_size = (max_size, int(h * max_size / w))
            img_show = img_show.resize(new_size)
        if save_path:
            dir_path = os.path.dirname(save_path)
            if dir_path and not os.path.isdir(dir_path):
                os.makedirs(dir_path)
            img_show.save(save_path)
            if return_pil_image:
                return img_show
        elif return_pil_image:
            return img_show
        else:
            buff = BytesIO()
            if format is None:
                format = "jpeg"
            if format.lower() == "jpg":
                format = "jpeg"
            img_show.save(buff, format=format, quality=90)
            return buff
