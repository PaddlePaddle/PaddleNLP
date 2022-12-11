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

import os
import json
import hashlib

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@article{Jaume2019FUNSDAD,
  title={FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
  author={Guillaume Jaume and H. K. Ekenel and J. Thiran},
  journal={2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
  year={2019},
  volume={2},
  pages={1-6}
}
"""

_DESCRIPTION = """\
https://guillaumejaume.github.io/FUNSD/
"""

_URL = "https://bj.bcebos.com/paddlenlp/datasets/funsd.tar.gz"


def _get_md5(string):
    """Get md5 value for string"""
    hl = hashlib.md5()
    hl.update(string.encode(encoding="utf-8"))
    return hl.hexdigest()


class FUNSDConfig(datasets.BuilderConfig):
    """funsd dataset config"""

    target_size: int = 1000
    max_size: int = 1000

    def __init__(self, **kwargs):

        super(FUNSDConfig, self).__init__(**kwargs)


class FUNSD(datasets.GeneratorBasedBuilder):
    """funsd dataset builder"""

    BUILDER_CONFIGS = [
        FUNSDConfig(
            name="funsd",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "name": datasets.Value("string"),
                    "page_no": datasets.Value("int32"),
                    "text": datasets.features.Sequence(datasets.Value("string")),
                    "bbox": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("int32"))),
                    "segment_bbox": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("int32"))),
                    "segment_id": datasets.features.Sequence(datasets.Value("int32")),
                    "image": datasets.Value("string"),
                    "width": datasets.Value("int32"),
                    "height": datasets.Value("int32"),
                    "md5sum": datasets.Value("string"),
                    "qas": datasets.features.Sequence(
                        {
                            "question_id": datasets.Value("int32"),
                            "question": datasets.Value("string"),
                            "answers": datasets.features.Sequence(
                                {
                                    "text": datasets.Value("string"),
                                    "answer_start": datasets.Value("int32"),
                                    "answer_end": datasets.Value("int32"),
                                }
                            ),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://guillaumejaume.github.io/FUNSD/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(dl_dir, "funsd", "train.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(dl_dir, "funsd", "dev.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(dl_dir, "funsd", "test.json")},
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("Generating examples from = {}".format(filepath))
        idx = 0
        with open(filepath, "r") as fin:
            for line in fin:
                data = json.loads(line)
                if "page_no" not in data:
                    data["page_no"] = 0
                for item in data["qas"]:
                    if "question_id" not in item:
                        item["question_id"] = -1
                data["md5sum"] = _get_md5(data["image"])
                yield idx, data
                idx += 1
