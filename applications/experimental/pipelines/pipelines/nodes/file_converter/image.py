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

from typing import List, Optional, Dict, Any, Union

import logging
import subprocess
from pathlib import Path
from paddleocr import PaddleOCR
try:
    from PIL.PpmImagePlugin import PpmImageFile
    from PIL import Image
except (ImportError, ModuleNotFoundError) as ie:
    from pipelines.utils.import_utils import _optional_component_not_installed

    _optional_component_not_installed(__name__, "ocr", ie)

from pipelines.nodes.file_converter import BaseConverter

logger = logging.getLogger(__name__)


class ImageToTextConverter(BaseConverter):

    def __init__(
        self,
        remove_numeric_tables: bool = False,
        valid_languages: Optional[List[str]] = ["eng"],
    ):
        """
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified here
                                (https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html)
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text. Run the following line of code to check available language packs:
                                # List of available languages
                                print(pytesseract.get_languages(config=''))
        """

        # save init parameters to enable export of component config as YAML
        self.set_config(remove_numeric_tables=remove_numeric_tables,
                        valid_languages=valid_languages)
        self.recognize = PaddleOCR(use_angle_cls=True, lang='ch')
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
        Extract text from image file using the pytesseract library (https://github.com/madmaze/pytesseract)

        :param file_path: path to image file
        :param meta: Optional dictionary with metadata that shall be attached to all resulting documents.
                     Can be any custom keys and values.
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
        pages = self._image_to_text(file_path)
        if remove_numeric_tables is None:
            remove_numeric_tables = self.remove_numeric_tables
        if valid_languages is None:
            valid_languages = self.valid_languages

        cleaned_pages = []
        for page in pages:
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
                        logger.debug(f"Removing line '{line}' from file")
                        continue
                cleaned_lines.append(line)
            cleaned_pages.append(page)

        if valid_languages:
            document_text = "".join(cleaned_pages)
            if not self.validate_language(document_text, valid_languages):
                logger.warning(
                    f"The language for image is not one of {valid_languages}. The file may not have "
                    f"been decoded in the correct text format.")
        documents = []
        for page in cleaned_pages:
            document = {"content": page, "meta": meta}
            documents.append(document)
        return documents

    def _image_to_text(self, img_path) -> List[str]:
        """
        Extract text from image path.

        :param image: input image path
        """
        img_path = str(img_path)
        result = self.recognize.ocr(img_path, cls=True)
        texts = []
        for line in result:
            texts.append(line[-1][0])
        texts = [''.join(texts)]
        return texts
