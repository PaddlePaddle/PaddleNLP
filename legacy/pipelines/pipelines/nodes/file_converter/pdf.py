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

import functools
import logging
import multiprocessing
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pypdf

try:
    from pdf2image import convert_from_path
except (ImportError, ModuleNotFoundError) as ie:
    from pipelines.utils.import_utils import _optional_component_not_installed

    _optional_component_not_installed(__name__, "ocr", ie)

from pipelines.nodes.file_converter import BaseConverter, ImageToTextConverter

logger = logging.getLogger(__name__)


def extract_pages(page_list, file_path):
    start = page_list[0]
    end = page_list[1]
    page_text = []
    pdf = pypdf.PdfReader(file_path)
    for index, page in enumerate(pdf.pages[start:end]):
        try:
            paragraphs = page.extract_text()
            paragraphs = paragraphs.encode("UTF-8", "ignore").decode("UTF-8")
            page_text.append(paragraphs)
        except Exception as e:
            logger.warning("Page %d of the file cannot be parsed correctly %s" % (index + start + 1, str(e)))
    return page_text


def run_process(pages, file_path, process_num=2):
    process_num = min(os.cpu_count(), process_num)
    pool = multiprocessing.Pool(process_num)
    extract_pages_c = functools.partial(extract_pages, file_path=file_path)
    result = pool.map_async(extract_pages_c, pages)
    pool.close()
    pool.join()
    return result.get()


class PDFToTextConverter(BaseConverter):
    def __init__(
        self,
        remove_numeric_tables: bool = False,
        language: str = "en",
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
        self.set_config(remove_numeric_tables=remove_numeric_tables, valid_languages=valid_languages)

        super().__init__(remove_numeric_tables=remove_numeric_tables, valid_languages=valid_languages)
        self.language = language

    def convert(
        self,
        file_path: Path,
        process_num: int = 20,
        meta: Optional[Dict[str, str]] = None,
        remove_numeric_tables: Optional[bool] = None,
        valid_languages: Optional[List[str]] = None,
        language: Optional[str] = "en",
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Extract text from a .pdf file using the pypdf library (https://pybrary.net/pyPdf/)

        :param file_path: Path to the .pdf file you want to convert
        :param process_num: Number of processes
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
        """
        pages = self._read_pdf(file_path, process_num=process_num)
        documents = []
        for page in pages:
            document = {"content": page, "content_type": "text", "meta": meta}
            documents.append(document)
        return documents

    def _read_pdf(self, file_path: Path, process_num: int) -> List[str]:
        """
        Extract pages from the pdf file at file_path.

        :param file_path: path of the pdf file
        :param layout: whether to retain the original physical layout for a page. If disabled, PDF pages are read in
                       the content stream order.
        ::param process_num: Number of processes
        """
        if process_num > os.cpu_count():
            logger.warning("The number of processes cannot exceed the number of cups")
            process_num = os.cpu_count()
        pdf = pypdf.PdfReader(file_path)
        page_length = len(pdf.pages)
        split_len = page_length // process_num
        if split_len == 0:
            split_len = page_length
        page_list = [i for i in range(0, page_length, split_len)]
        if page_length > page_list[-1]:
            page_list.append(page_length)
        page_combination = [(start, end) for start, end in zip(page_list, page_list[1:])]
        page_text = run_process(page_combination, file_path, process_num)
        page_text_all = []
        for item in page_text:
            page_text_all.extend(item)
        return page_text_all


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
        self.image_2_text = ImageToTextConverter(remove_numeric_tables, valid_languages)

        # save init parameters to enable export of component config as YAML
        self.set_config(remove_numeric_tables=remove_numeric_tables, valid_languages=valid_languages)
        super().__init__(remove_numeric_tables=remove_numeric_tables, valid_languages=valid_languages)

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
                temp_img = tempfile.NamedTemporaryFile(
                    dir=os.path.dirname(os.path.realpath(__file__)), suffix=".jpeg", delete=False
                )
                image.save(temp_img.name)
                pages.append(self.image_2_text.convert(temp_img.name)[0]["content"])
                temp_img.close()
                os.remove(temp_img.name)
        except Exception as exception:
            logger.error(f"File {file_path} has an error \n {exception}")

        raw_text = "\f".join(pages)
        document = {"content": raw_text, "meta": meta}

        return [document]
