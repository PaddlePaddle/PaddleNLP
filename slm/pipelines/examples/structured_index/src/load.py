# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import logging
import os
from typing import Dict, List


class Loader:
    @classmethod
    def load_file(cls, filepath: str) -> Dict:
        """
        Load a file based on its extension using the appropriate loader method.

        Args:
            filepath (str): The path to the file to be loaded.

        Returns:
            Dict: A dictionary containing the loaded document data.
            None: If the file type is unsupported.
        """
        loader = {".pdf": cls.load_pdf, ".json": cls.load_json, ".html": cls.load_html}
        _, ext = os.path.splitext(os.path.basename(filepath))
        if ext in loader:
            doc = loader[ext](filepath=filepath)
            return doc
        else:
            logging.warning(f"Unsupported file type: {filepath}")
            return None

    @classmethod
    def load_dir(cls, input_dir: str, output_dir: str, save: bool = True):
        """
        Load all files in a directory and optionally save their content as JSON files.

        Args:
            input_dir (str): The directory containing the files to be loaded.
            output_dir (str): The directory where the loaded content will be saved as JSON files.
            save (bool, optional): Whether to save the loaded content. Defaults to True.
        """
        for root, _, files in os.walk(input_dir):
            for file in files:
                filepath = os.path.join(root, file)
                filename, _ = os.path.splitext(os.path.basename(file))
                file_output_dir = os.path.join(output_dir, f"{filename}.json")
                if os.path.exists(file_output_dir):
                    if os.path.getsize(file_output_dir) > 0:
                        logging.info(f"File already exists: {file_output_dir}")
                        continue
                doc = cls.load_file(filepath=filepath)
                if save and doc is not None:
                    with open(file_output_dir, "w") as f:
                        json.dump(doc, f, ensure_ascii=False, indent=4)
                    logging.info(f"Saved to: {file_output_dir}")

    @classmethod
    def load_pdf(cls, filepath: str) -> Dict:
        """
        Load a PDF file, extract images from each page, perform OCR on the images, and return the extracted text content.

        Args:
            filepath (str): The path to the PDF file to be loaded.

        Returns:
            Dict: A dictionary containing the extracted text content from the PDF.
        """

        logging.info(f"Loading PDF: {filepath}")

        from paddleocr import PaddleOCR

        # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
        # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
        PAGE_NUM = 10  # 将识别页码前置作为全局，防止后续打开pdf的参数和前文识别参数不一致 / Set the recognition page number
        # ocr = PaddleOCR(use_angle_cls=True, lang="ch", page_num=PAGE_NUM)  # need to run only once to download and load model into memory
        ocr = PaddleOCR(
            use_angle_cls=True, lang="ch", page_num=PAGE_NUM, use_gpu=1
        )  # 如果需要使用GPU，请取消此行的注释 并注释上一行 / To Use GPU,uncomment this line and comment the above one.
        result = ocr.ocr(filepath, cls=True)

        doc = {"source": filepath, "nodes": []}
        for idx in range(len(result)):
            res = result[idx]
            if res is None:
                # 识别到空页就跳过，防止程序报错 / Skip when empty result detected to avoid TypeError:NoneType
                logging.info(f"Empty page {idx+1} detected in {filepath}, skip it.")
                continue
            for line in res:
                assert isinstance(line, list) and len(line) == 2
                assert isinstance(line[1], tuple) and len(line[1]) == 2
                assert isinstance(line[1][0], str) and isinstance(line[1][1], float)
                doc["nodes"].append(
                    {
                        "id": len(doc["nodes"]),
                        "content": line[1][0],
                        "confidence": line[1][1],
                    }
                )
        return doc

    @classmethod
    def load_json(cls, filepath: str) -> dict:
        with open(filepath, "r") as f:
            data = json.load(f)
        return data

    @classmethod
    def load_html(cls, filepath: str) -> dict:
        """
        Load an HTML file, extract text content, and return it as a structured dictionary.

        Args:
            filepath (str): The path to the HTML file to be loaded.

        Returns:
            dict: A dictionary containing the extracted text content from the HTML file.
        """
        from bs4 import BeautifulSoup

        doc = {"source": filepath, "nodes": []}
        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        html_text = soup.get_text()

        node_list: List[Dict] = []
        for t in html_text.splitlines():
            content = t.strip()
            if len(content) <= 0:
                continue
            node_list.append(
                {
                    "id": len(node_list),
                    "content": content,
                }
            )
        doc["nodes"] = node_list

        return doc
