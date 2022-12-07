# coding=utf-8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3

import json
import os

import datasets
from datasets.tasks import QuestionAnsweringExtractive

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """\
Duconv is a chinese conversation \
dataset, designed to evaluate the dialogue models.
"""

_URL = "https://bj.bcebos.com/paddlenlp/datasets/DuConv.zip"


class DuconvConfig(datasets.BuilderConfig):
    """BuilderConfig for Duconv."""

    def __init__(self, **kwargs):
        """BuilderConfig for Duconv.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DuconvConfig, self).__init__(**kwargs)


class Duconv(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        DuconvConfig(
            name="DuConv",
            version=datasets.Version("1.0.0", ""),
            description=_DESCRIPTION,
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "goal": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                    "knowledge": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                    "conversation": datasets.Sequence(datasets.Value("string")),
                    "history": datasets.Sequence(datasets.Value("string")),
                    "response": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://arxiv.org/pdf/1906.05572.pdf",
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name="train",
                gen_kwargs={
                    "filepath": os.path.join(dl_dir, "DuConv", "train.txt"),
                },
            ),
            datasets.SplitGenerator(
                name="dev",
                gen_kwargs={
                    "filepath": os.path.join(dl_dir, "DuConv", "dev.txt"),
                },
            ),
            datasets.SplitGenerator(
                name="test_1",
                gen_kwargs={
                    "filepath": os.path.join(dl_dir, "DuConv", "test_1.txt"),
                },
            ),
            datasets.SplitGenerator(
                name="test_2",
                gen_kwargs={
                    "filepath": os.path.join(dl_dir, "DuConv", "test_2.txt"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, "r", encoding="utf-8") as fin:
            for line in fin:
                duconv = json.loads(line)

                goal = duconv["goal"] if "goal" in duconv.keys() else [[]]
                knowledge = duconv["knowledge"] if "knowledge" in duconv.keys() else [[]]
                conversation = duconv["conversation"] if "conversation" in duconv.keys() else []
                history = duconv["history"] if "history" in duconv.keys() else []
                response = duconv["response"] if "response" in duconv.keys() else ""

                yield key, {
                    "id": str(key),
                    "goal": goal,
                    "knowledge": knowledge,
                    "conversation": conversation,
                    "history": history,
                    "response": response,
                }
                key += 1
