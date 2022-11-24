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

import re
import os
import base64
import mimetypes
from PIL import Image
from io import BytesIO
import requests
import numpy as np

from .log import logger
from .image_utils import np2base64


class DocParser(object):
    """DocParser"""

    def __init__(self,
                 ocr_model_config='PP-OCRv3',
                 layout_analysis=False,
                 pdf_parser_config=None,
                 use_gpu=None,
                 device_id=None):
        self.ocr_model_config = ocr_model_config
        self.use_angle_cls = False
        self.layout_analysis = layout_analysis
        if isinstance(ocr_model_config, dict):
            self.use_angle_cls = ocr_model_config.get('use_angle_cls', False)
        self.pdf_parser_config = pdf_parser_config
        self.ocr_infer_model = None
        self.use_gpu = use_gpu
        self.device_id = device_id

    def parse(self,
              doc,
              keep_whitespace=False,
              expand_to_a4_size=False,
              return_ocr_result=True):
        """
        parse
        """
        doc_type = mimetypes.guess_type(doc['doc'])[0]

        if not doc_type or doc_type.startswith("image"):
            image = self.read_image(doc['doc'])
        elif doc_type == "application/pdf":
            image = self.read_pdf(doc['doc'])
        offset_x, offset_y = 0, 0
        if expand_to_a4_size:
            image, offset_x, offset_y = self.expand_image_to_a4_size(
                image, center=True)
        img_w, img_h = image.shape[1], image.shape[0]
        doc['image'] = np2base64(image)
        doc['offset_x'] = offset_x
        doc['offset_y'] = offset_y
        doc['img_w'] = img_w
        doc['img_h'] = img_h
        if return_ocr_result:
            ocr_result = self.ocr(image, keep_whitespace=keep_whitespace)
            doc['bbox'] = ocr_result
        return doc

    def __call__(self, *args, **kwargs):
        """
        Call parse
        """
        return self.parse(*args, **kwargs)

    def ocr(self, image, det=True, rec=True, cls=None, keep_whitespace=True):
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
                if not keep_whitespace:
                    text = text.replace(' ', '')
                layout.append((box, text, segment[1][1]))
        else:
            layout_result = self.layout_analysis_engine(image)
            for region in layout_result:
                if region['type'] != "table":
                    ocr_result = region['res']
                    for segment in ocr_result:
                        box = segment['text_region']
                        box = _get_box(box)
                        text = segment['text']
                        if not keep_whitespace:
                            text = text.replace(' ', '')
                        layout.append((box, text, segment['confidence']))
                else:
                    table_result = region['res']
                    html = table_result['html']
                    cell_bbox = table_result['cell_bbox']

                    table_list = []
                    lines = re.findall('<tr>(.*?)</tr>', html)
                    for line in lines:
                        table_list.extend(re.findall('<td>(.*?)</td>', line))
                        table_list.extend(
                            re.findall('<td colspan="2">(.*?)</td>', line))
                    for cell_box, text in zip(cell_bbox, table_list):
                        box = [
                            cell_box[0], cell_box[1], cell_box[4], cell_box[5]
                        ]
                        layout.append((box, text.replace(" ", "")))
        return layout

    @classmethod
    def _get_buffer(self, data, file_like=False):
        buff = None
        if len(data) < 1024:
            if os.path.exists(data):
                buff = open(data, 'rb').read()
            elif data.startswith("http://") or data.startswith("https://"):
                resp = requests.get(data, stream=True)
                if not resp.ok:
                    raise RuntimeError(
                        "Failed to download the file from {}".format(data))
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
                "Need PyMuPDF to process pdf input. "
                "Please install module by: python3 -m pip install pymupdf")
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
            logger.warning(
                "Currently only parse the first page for PDF input with more than one page."
            )

        page = pdf_doc.load_page(0)
        image = np.array(self.get_page_image(page).convert("RGB"))
        return image

    @classmethod
    def get_page_image(self, page):
        """
        get page image
        """
        pix = page.get_pixmap()
        image_buff = pix.pil_tobytes('jpeg', optimize=True)
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
            if isinstance(self.ocr_model_config, dict):
                self.ocr_infer_model = PaddleOCR(**self.ocr_model_config)
            else:
                self.ocr_infer_model = PaddleOCR(
                    ocr_version=self.ocr_model_config, show_log=False)
        else:
            try:
                from paddleocr import PPStructure
            except ImportError:
                raise RuntimeError(
                    "Need paddleocr to process image input. "
                    "Please install module by: python3 -m pip install paddleocr"
                )
            self.layout_analysis_engine = PPStructure(table=True,
                                                      ocr=True,
                                                      show_log=False)

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
                exp_img = np.zeros((h, offset_x, 3), dtype='uint8')
                exp_img.fill(255)
                image = np.hstack([exp_img, image, exp_img])
            else:
                exp_img = np.zeros((h, exp_w, 3), dtype='uint8')
                exp_img.fill(255)
                image = np.hstack([image, exp_img])
        elif h * 1.0 / w <= 1.40:
            exp_h = int(w * 1.414 - h)
            if center:
                offset_y = int(exp_h / 2)
                exp_img = np.zeros((offset_y, w, 3), dtype='uint8')
                exp_img.fill(255)
                image = np.vstack([exp_img, image, exp_img])
            else:
                exp_img = np.zeros((exp_h, w, 3), dtype='uint8')
                exp_img.fill(255)
                image = np.vstack([image, exp_img])
        return image, offset_x, offset_y
