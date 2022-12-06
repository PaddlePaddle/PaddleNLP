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

import os
import logging
from typing import List, Dict
import numpy as np
import base64
import sys
sys.path.insert(1, "./../..")
sys.path.insert(2, "./../../..")

import paddle
from paddleocr import PaddleOCR
from pathlib import Path
from paddlenlp.taskflow.utils import download_file

from pipelines.nodes.base import BaseComponent

logger = logging.getLogger(__name__)


class SentaProcessor(BaseComponent):
    """
    Read and preprocess texts that you wanna perform sentiment analysis.
    """
    return_no_answers: bool
    outgoing_edges = 1
    query_count = 0
    query_time = 0

    def __init__(self, max_examples: int = None):
        """
        Init Senta Preprocessor.
        :param max_examples: Maximum amount of examples to process. if you set to be None, it will keep all examples to analyze.
        """
        self.max_examples = max_examples

    def _check_input_params(self, inputs):
        if not isinstance(inputs, dict):
            raise TypeError("a expected dict as input, but received {}!".format(type(inputs)))
        if "file_path" not in inputs:
            raise ValueError("a file path is needed, which you wanna perform sentiment analysis, you can set it by `file_path`.")
        if not os.path.exists(inputs["file_path"]):
            raise ValueError("the file does not exist: {}".format(inputs["file_path"]))
        if "save_path" in inputs and not isinstance(inputs["save_path"], str):
            raise TypeError("a str expected for save_path, but received {}!".format(type(inputs["save_path"])))

    def _read_text_file(self, file_path):
        examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                example = line.strip()
                examples.append(example)
        return examples

    def run(self, meta: dict):
        # check the param meta, file_path need to be input as key.
        self._check_input_params(meta)
        # read texts
        examples = self._read_text_file(meta["file_path"])
        examples = examples[:self.max_examples]

        output = {"examples": examples}
        if "save_path" in meta and meta["save_path"]:
            output["save_path"] = meta["save_path"]

        return output, "output_1"
