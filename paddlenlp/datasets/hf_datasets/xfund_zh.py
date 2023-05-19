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
@inproceedings{xu-etal-2022-xfund,
    title = "{XFUND}: A Benchmark Dataset for Multilingual Visually Rich Form Understanding",
    author = "Xu, Yiheng  and
      Lv, Tengchao  and
      Cui, Lei  and
      Wang, Guoxin  and
      Lu, Yijuan  and
      Florencio, Dinei  and
      Zhang, Cha  and
      Wei, Furu",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.253",
    doi = "10.18653/v1/2022.findings-acl.253",
    pages = "3214--3224",
    abstract = "Multimodal pre-training with text, layout, and image has achieved SOTA performance for visually rich document understanding tasks recently, which demonstrates the great potential for joint learning across different modalities. However, the existed research work has focused only on the English domain while neglecting the importance of multilingual generalization. In this paper, we introduce a human-annotated multilingual form understanding benchmark dataset named XFUND, which includes form understanding samples in 7 languages (Chinese, Japanese, Spanish, French, Italian, German, Portuguese). Meanwhile, we present LayoutXLM, a multimodal pre-trained model for multilingual document understanding, which aims to bridge the language barriers for visually rich document understanding. Experimental results show that the LayoutXLM model has significantly outperformed the existing SOTA cross-lingual pre-trained models on the XFUND dataset. The XFUND dataset and the pre-trained LayoutXLM model have been publicly available at https://aka.ms/layoutxlm.",
}
"""

_DESCRIPTION = """\
https://github.com/doc-analysis/XFUND
"""

_URL = "https://bj.bcebos.com/paddlenlp/datasets/xfund_zh.tar.gz"


def _get_md5(string):
    """Get md5 value for string"""
    hl = hashlib.md5()
    hl.update(string.encode(encoding="utf-8"))
    return hl.hexdigest()


class XFUNDZhConfig(datasets.BuilderConfig):
    """xfund_zh dataset config"""

    target_size: int = 1000
    max_size: int = 1000

    def __init__(self, **kwargs):

        super(XFUNDZhConfig, self).__init__(**kwargs)


class XFUNDZh(datasets.GeneratorBasedBuilder):
    """xfund_zh dataset builder"""

    BUILDER_CONFIGS = [
        XFUNDZhConfig(
            name="xfund_zh",
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
            homepage="https://github.com/doc-analysis/XFUND",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(dl_dir, "xfund_zh", "train.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(dl_dir, "xfund_zh", "dev.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(dl_dir, "xfund_zh", "test.json")},
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
