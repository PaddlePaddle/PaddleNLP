# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 deepset GmbH. All Rights Reserved.
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

from typing import List, Optional, Dict, Any

import os
import logging
import tempfile
import time
import subprocess
from pathlib import Path
import pdfplumber
import sys
import hashlib
import urllib.parse
import requests, json
from datetime import datetime

from pipelines.nodes.file_converter import BaseConverter, ImageToTextConverter

logger = logging.getLogger(__name__)


class PDFToTextConverter(BaseConverter):

    def __init__(
        self,
        remove_numeric_tables: bool = False,
        valid_languages: Optional[List[str]] = None,
    ):
        """
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        """
        # save init parameters to enable export of component config as YAML
        self.set_config(remove_numeric_tables=remove_numeric_tables,
                        valid_languages=valid_languages)

        super().__init__(remove_numeric_tables=remove_numeric_tables,
                         valid_languages=valid_languages)

    def convert(
        self,
        file_path: Path,
        meta: Optional[Dict[str, str]] = None,
        remove_numeric_tables: Optional[bool] = None,
        valid_languages: Optional[List[str]] = None,
        encoding: Optional[str] = "Latin1",
    ) -> List[Dict[str, Any]]:
        """
        Extract text from a .pdf file using the pdftotext library (https://www.xpdfreader.com/pdftotext-man.html)

        :param file_path: Path to the .pdf file you want to convert
        :param meta: Optional dictionary with metadata that shall be attached to all resulting documents.
                     Can be any custom keys and values.
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        :param encoding: Encoding that will be passed as -enc parameter to pdftotext. "Latin 1" is the default encoding
                         of pdftotext. While this works well on many PDFs, it might be needed to switch to "UTF-8" or
                         others if your doc contains special characters (e.g. German Umlauts, Cyrillic characters ...).
                         Note: With "UTF-8" we experienced cases, where a simple "fi" gets wrongly parsed as
                         "xef\xac\x81c" (see test cases). That's why we keep "Latin 1" as default here.
                         (See list of available encodings by running `pdftotext -listenc` in the terminal)
        """

        pages = self._read_pdf(file_path, layout=False, encoding=encoding)
        if remove_numeric_tables is None:
            remove_numeric_tables = self.remove_numeric_tables
        if valid_languages is None:
            valid_languages = self.valid_languages

        cleaned_pages = []
        for page in pages:
            # pdftotext tool provides an option to retain the original physical layout of a PDF page. This behaviour
            # can be toggled by using the layout param.
            #  layout=True
            #      + table structures get retained better
            #      - multi-column pages(eg, research papers) gets extracted with text from multiple columns on same line
            #  layout=False
            #      + keeps strings in content stream order, hence multi column layout works well
            #      - cells of tables gets split across line
            #
            #  Here, as a "safe" default, layout is turned off.
            lines = page.splitlines()
            cleaned_lines = []
            for line in lines:
                words = line.split()
                digits = [
                    word for word in words if any(i.isdigit() for i in word)
                ]

                # remove lines having > 40% of words as digits AND not ending with a period(.)
                if remove_numeric_tables:
                    if words and len(digits) / len(
                            words) > 0.4 and not line.strip().endswith("."):
                        logger.debug(f"Removing line '{line}' from {file_path}")
                        continue
                cleaned_lines.append(line)
            cleaned_pages.extend(cleaned_lines)

        if valid_languages:
            document_text = "".join(cleaned_pages)
            if not self.validate_language(document_text, valid_languages):
                logger.warning(
                    f"The language for {file_path} is not one of {valid_languages}. The file may not have "
                    f"been decoded in the correct text format.")

        documents = []
        for page in cleaned_pages:
            document = {"content": page, "content_type": "text", "meta": meta}
            documents.append(document)
        return documents

    def _read_pdf(self,
                  file_path: Path,
                  layout: bool,
                  encoding: Optional[str] = "Latin1") -> List[str]:
        """
        Extract pages from the pdf file at file_path.

        :param file_path: path of the pdf file
        :param layout: whether to retain the original physical layout for a page. If disabled, PDF pages are read in
                       the content stream order.
        """
        pdf = pdfplumber.open(file_path)
        page_text = []
        for page in pdf.pages:
            paragraphs = page.extract_text()
            page_text.append(paragraphs)
        return page_text


class PDFluxToTextConverter(BaseConverter):

    def __init__(
        self,
        username: Optional[str] = "user",
        secret_key: Optional[str] = "bIokCvUdPVnz",
        remove_numeric_tables: bool = False,
        valid_languages: Optional[List[str]] = None,
    ):
        """
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        """
        # Save init parameters to enable export of component config as YAML
        self.set_config(remove_numeric_tables=remove_numeric_tables,
                        valid_languages=valid_languages)

        super().__init__(remove_numeric_tables=remove_numeric_tables,
                         valid_languages=valid_languages)
        self.username = username
        self.secret_key = secret_key
        self.base_url = 'http://saas.pdflux.com/api/v1/saas'

    def convert(
        self,
        file_path: Path,
        meta: Optional[Dict[str, str]] = None,
        remove_numeric_tables: Optional[bool] = None,
        valid_languages: Optional[List[str]] = None,
        encoding: Optional[str] = "Latin1",
    ) -> List[Dict[str, Any]]:
        """
        Extract text from a .pdf file using the pdflux API (https://pdflux.com/), the API will return a JSON file, eg:

        {
            "document": [
                {
                    "id": 2022,
                    "uuid": "2afdca4a-5c65-11eb-a1f8-00163e055917", // 文件uuid
                    "parsed": 2,
                    "filename": "中国广核电力股份有限公司主体与2019年度第一期中期票据信用评级报告（中诚信国际）.pdf", // 文件名
                    "created_utc": 1611287515, // 创建时间
                    "updated_utc": 1611287688, // 修改事件
                    "exceptions": null // 报错信息
                }
            ],
            "pdf_page": [
                {
                    "id": 10379,
                    "did": 2022,
                    "page": 0, // 页面序号、页码
                    "meta": {
                    "width": 595, // 页面宽度
                    "height": 842, // 页面高度
                    "page_type": null,
                    "page_prob": null,
                    "is_image": true // 是否是扫描件或图片
                    },
                    "created_utc": 1611287679,
                    "updated_utc": 1611287679
                },
                ... // ...表示省略
            ],
            "pdf_elements": [
                {
                    "page": 1, // 页面序号、页码
                    "elements": [
                    {
                        "page": 1, // 页面序号、页码
                        "text": "中诚信国呩 CCXI-20182331D-01", // 文字内容
                        "index": 0,
                        "element_type": "page_headers" // 元素块类型：页眉
                    },
                    {
                        "page": 1,
                        "text": "中国广核电力股份有限公司2019年度第一期中期票据信用评级报告",
                        "index": 1,
                        "syllabus": 1, // 和目录的对应关系
                        "element_type": "paragraphs" // 元素块类型：段落
                    },
                    {
                        "unit": "", // 表格单位 
                        "cells": { // 表格单元格
                            "0_0": { // 
                                "value": "发行主体" // 单元格内的文字内容
                            },
                            "0_1": { // 单元格位置信息：第一个“0”代表的是行数，第二个”0“代表的是列数，”0_0"代表第一个单元格
                                "value": "中国卜核*力股份有限公司"
                            },
                            ... // 表示省略
                        },
                        "title": "中国广核电力股份有限公司2019年度第一期中期票据信用评级报告",
                        "merged": [ // 单元格合并信息
                            [
                                [0,1],[0,2],[0,3],[0,4] // 表示这4个单元格合并
                            ],
                            [
                                [1,1],[1,2],[1,3],[1,4] // 表示这4个单元格合并
                            ],
                            ... // 表示省略
                        ],
                        "element_type": "tables", // 元素块类型：表格
                        "page": 1, // 页码
                        "index": 2 // 在页面中出现的顺序
                    },
                    {
                        "page": 1,
                        "text": "www.ccxi.com.cn 中国广核电力股份有限公司2019年度第一期中期票据信用评级报告",
                        "index": 3,
                        "element_type": "page_footers" // 元素块类型：页脚
                    }
                    ]
                },
                {
                    "page": 2,
                    "elements": [
                        {
                            "data": "iVBORw0KGgoA......", // 图片内容，Base64格式
                            "page": 2, // 页码
                            "index": 0, // 在页面中出现的顺序
                            "element_type": "images"  // 元素块类型：图片
                        },
                        {
                            "page": 2,
                            "text": "关 注",
                            "index": 1,
                            "element_type": "paragraphs",
                        },
                        ...
                    ]
                },
            ...
            ],
            "syllabus": { // 目录
                "index": -1, // 目录根节点 
                "children": [ // 子节点
                    {
                        "page": 1,
                        "etype": "paragraphs", 
                        "index": 1, // 目录序号 
                        "level": 1, // 目录层级
                        "range": [ // 当前目录包含的元素块
                            1,10
                        ],
                        "title": "中国广核电力股份有限公司2019年度第一期中期票据信用评级报告", // 目录标题 
                        "parent": -1, // 父节点
                        "element": 1, // 目录与元素块的对应关系
                        "children": [ // 子节点
                        {
                            "page": 2,
                            "etype": "paragraphs",
                            "index": 2,
                            "level": 2,
                            "range": [
                                5,10
                            ],
                            "title": "关 注",
                            "parent": 1,
                            "element": 5,
                            "children": []
                        }
                        ]
                    },
                    {
                        "page": 3,
                        "etype": "paragraphs",
                        "index": 3,
                        "level": 1,
                        "range": [
                            10,20
                        ],
                        "title": "声明",
                        "parent": -1,
                        "element": 10,
                        "children": []
                    },
                    ...
                ]
            }        
            }


        :param file_path: Path to the .pdf file you want to convert
        :param meta: Optional dictionary with metadata that shall be attached to all resulting documents.
                     Can be any custom keys and values.
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        :param encoding: Encoding that will be passed as -enc parameter to pdftotext. "Latin 1" is the default encoding
                         of pdftotext. While this works well on many PDFs, it might be needed to switch to "UTF-8" or
                         others if your doc contains special characters (e.g. German Umlauts, Cyrillic characters ...).
                         Note: With "UTF-8" we experienced cases, where a simple "fi" gets wrongly parsed as
                         "xef\xac\x81c" (see test cases). That's why we keep "Latin 1" as default here.
                         (See list of available encodings by running `pdftotext -listenc` in the terminal)
        """
        # Upload files to pdflux server
        upload_url = '{}/upload?user={}&force_updata=true'.format(
            self.base_url, self.username)
        upload_url = self.encode_url(upload_url, 'pdflux', self.secret_key)
        data = {'file': open(file_path, 'rb')}
        cleaned_pages = []
        try:
            result = requests.post(upload_url, files=data)
            uuid = result.json()['data']['uuid']
            logger.info('Upload file uuid is {}'.format(uuid))
            while True:
                time.sleep(1)
                # Get file status for file processing 2 represents the files processing has been finished
                statue_url = '{}/document/{}?user={}'.format(
                    self.base_url, uuid, self.username)
                statue_url = self.encode_url(statue_url, 'pdflux',
                                             self.secret_key)
                result = requests.get(statue_url).json()
                if (result['data']['parsed'] == 2):
                    # Get the parsing result from pdflux
                    result_url = '{}/document/{}/pdftables?user={}'.format(
                        self.base_url, uuid, self.username)
                    down_url = self.encode_url(result_url, 'pdflux',
                                               self.secret_key)
                    down_res = requests.get(url=down_url).json()
                    # Extract text from pdf_elements for general text
                    for pages in down_res['pdf_elements']:
                        for page in pages['elements']:
                            if (page['element_type'] == 'paragraphs'):
                                cleaned_pages.append(page['text'])
                    # Extract text from syllabus for text with titles, subtitles
                    if ('children' in down_res['syllabus']):
                        for children in down_res['syllabus']['children']:
                            list_data = []
                            list_data = self.get_target_varlue(
                                'title', children, list_data)
                            cleaned_pages += list_data
                    break

        except Exception as e:
            logger.error(
                f"File {file_path} has an error \n {e}, Please check your pdflux api"
            )
        documents = []
        for page in cleaned_pages:
            document = {"content": page, "content_type": "text", "meta": meta}
            documents.append(document)
        return documents

    def revise_url(self, url, extra_params=None, excludes=None):
        extra_params = extra_params or {}
        excludes = excludes or []
        main_url, query = urllib.parse.splitquery(url)
        params = urllib.parse.parse_qs(query) if query else {}
        params.update(extra_params)
        keys = list(params.keys())
        keys.sort()
        params_strings = []
        for key in keys:
            if key in excludes:
                continue
            values = params[key]
            if isinstance(values, list):
                values.sort()
                params_strings.extend([
                    "{}={}".format(key, urllib.parse.quote(str(value)))
                    for value in values
                ])
            else:
                params_strings.append("{}={}".format(
                    key, urllib.parse.quote(str(values))))

        return "{}?{}".format(
            main_url, "&".join(params_strings)) if params_strings else main_url

    def generate_timestamp(self, ):
        delta = datetime.utcnow() - datetime.utcfromtimestamp(0)
        return int(delta.total_seconds())

    def _generate_token(self,
                        url,
                        app_id,
                        secret_key,
                        extra_params=None,
                        timestamp=None):
        url = self.revise_url(url,
                              extra_params=extra_params,
                              excludes=["_token", "_timestamp"])
        timestamp_now = timestamp or self.generate_timestamp()
        source = "{}#{}#{}#{}".format(url, app_id, secret_key, timestamp_now)
        token = hashlib.md5(source.encode()).hexdigest()
        return token

    def encode_url(self, url, app_id, secret_key, params=None, timestamp=None):
        timestamp = timestamp or self.generate_timestamp()
        token = self._generate_token(url, app_id, secret_key, params, timestamp)
        extra_params = {'_timestamp': timestamp, '_token': token}
        extra_params.update(params or {})
        url = self.revise_url(url, extra_params=extra_params)
        return url

    def get_target_varlue(self, key, dic, tmp_list):
        """
        :param key: the key that you want to get
        :param dic: JSON data
        :param tmp_list: Use to Store the value of the target key
        :return: list
        """
        if not isinstance(dic, dict) or not isinstance(tmp_list,
                                                       list):  # 对传入数据进行格式校验
            return 'argv[1] not an dict or argv[-1] not an list '
        if key in dic.keys():
            tmp_list.append(dic[key])  # 传入数据存在则存入tmp_list

        for value in dic.values():  # 传入数据不符合则对其value值进行遍历
            if isinstance(value, dict):
                self.get_target_varlue(key, value,
                                       tmp_list)  # 传入数据的value值是字典，则直接调用自身
            elif isinstance(value, (list, tuple)):
                self.__get_value(key, value,
                                 tmp_list)  # 传入数据的value值是列表或者元组，则调用_get_value
        return tmp_list

    def __get_value(self, key, val, tmp_list):
        for val_ in val:
            if isinstance(val_, dict):
                self.get_target_varlue(
                    key, val_, tmp_list)  # 传入数据的value值是字典，则调用get_target_value
            elif isinstance(val_, (list, tuple)):
                self.__get_value(key, val_,
                                 tmp_list)  # 传入数据的value值是列表或者元组，则调用自身


class PDFToTextOCRConverter(BaseConverter):

    def __init__(
        self,
        remove_numeric_tables: bool = False,
        valid_languages: Optional[List[str]] = ["eng"],
    ):
        """
        Extract text from image file using the pytesseract library (https://github.com/madmaze/pytesseract)

        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages supported by tessarect
                                (https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html).
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        """
        # init image to text instance
        self.image_2_text = ImageToTextConverter(remove_numeric_tables,
                                                 valid_languages)

        # save init parameters to enable export of component config as YAML
        self.set_config(remove_numeric_tables=remove_numeric_tables,
                        valid_languages=valid_languages)
        super().__init__(remove_numeric_tables=remove_numeric_tables,
                         valid_languages=valid_languages)

    def convert(
        self,
        file_path: Path,
        meta: Optional[Dict[str, str]] = None,
        remove_numeric_tables: Optional[bool] = None,
        valid_languages: Optional[List[str]] = None,
        encoding: Optional[str] = "utf-8",
    ) -> List[Dict[str, Any]]:
        """
        Convert a file to a dictionary containing the text and any associated meta data.

        File converters may extract file meta like name or size. In addition to it, user
        supplied meta data like author, url, external IDs can be supplied as a dictionary.

        :param file_path: path of the file to convert
        :param meta: dictionary of meta data key-value pairs to append in the returned document.
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        :param encoding: Select the file encoding (default is `utf-8`)
        """
        pages = []
        try:
            images = convert_from_path(file_path)
            for image in images:
                temp_img = tempfile.NamedTemporaryFile(dir=os.path.dirname(
                    os.path.realpath(__file__)),
                                                       suffix=".jpeg")
                image.save(temp_img.name)
                pages.append(
                    self.image_2_text.convert(temp_img.name)[0]["content"])
        except Exception as exception:
            logger.error(f"File {file_path} has an error \n {exception}")

        raw_text = "\f".join(pages)
        document = {"content": raw_text, "meta": meta}

        return [document]
