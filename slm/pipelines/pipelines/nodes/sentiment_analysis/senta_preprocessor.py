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

from pipelines.nodes.base import BaseComponent


class SentaProcessor(BaseComponent):
    """
    Read and preprocess texts that you wanna perform sentiment analysis.
    """

    outgoing_edges = 1

    def __init__(self, max_examples: int = -1):
        """
        Init Senta Preprocessor.
        :param max_examples: Maximum amount of examples to process. if you set to be -1, it will keep all examples to analyze.
        """
        self.max_examples = max_examples

    def _check_input_params(self, inputs):
        if not isinstance(inputs, dict):
            raise TypeError("a expected dict as input, but received {}!".format(type(inputs)))
        if "file_path" not in inputs:
            raise ValueError(
                "a file path is needed, which you wanna perform sentiment analysis, you can set it by `file_path`."
            )
        if not os.path.exists(inputs["file_path"]):
            raise ValueError("the file does not exist: {}".format(inputs["file_path"]))

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
        if self.max_examples != -1:
            examples = examples[: self.max_examples]
        # define output for SentaProcessor
        sr_file_name = "sr_" + os.path.basename(meta["file_path"]).split(".")[0] + ".json"
        sr_save_path = os.path.join(os.path.dirname(meta["file_path"]), "images", sr_file_name)
        output = {"examples": examples, "sr_save_path": sr_save_path}

        return output, "output_1"
