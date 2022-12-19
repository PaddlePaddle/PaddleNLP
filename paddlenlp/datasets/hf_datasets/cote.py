# coding=utf-8
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
"""COTE: Chinese Opinion Target Extraction."""

import csv
import os
import textwrap
import numpy as np

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@inproceedings{li2018character,
  title={Character-based bilstm-crf incorporating pos and dictionaries for chinese opinion target extraction},
  author={Li, Yanzeng and Liu, Tingwen and Li, Diying and Li, Quangang and Shi, Jinqiao and Wang, Yanqiu},
  booktitle={Asian Conference on Machine Learning},
  pages={518--533},
  year={2018},
  organization={PMLR}
}
"""

_DESCRIPTION = """\
COTE, a dataset for Opinion target extraction (OTE) for sentiment analysis, which aims to extract target of a given text. This dataset covers data crawled on Baidu, Dianping, and Mafengwo.
More information refer to https://www.luge.ai/#/luge/dataDetail?id=19.
"""

_COTE_URLs = {
    # pylint: disable=line-too-long
    "bd": "https://paddlenlp.bj.bcebos.com/datasets/COTE-BD.zip",
    "mfw": "https://paddlenlp.bj.bcebos.com/datasets/COTE-MFW.zip",
    "dp": "https://paddlenlp.bj.bcebos.com/datasets/COTE-DP.zip",
    # pylint: enable=line-too-long
}


class COTEConfig(datasets.BuilderConfig):
    """BuilderConfig for COTE."""

    def __init__(self, data_url=None, data_dir=None, **kwargs):
        """BuilderConfig for COTE.

        Args:
          data_url: `string`, url to download the zip file.
          data_dir: `string`, the path to the folder containing the tsv files in the downloaded zip.
          **kwargs: keyword arguments forwarded to super.
        """
        super(COTEConfig, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir


class COTE(datasets.GeneratorBasedBuilder):
    """COTE: Chinese Opinion Target Extraction."""

    BUILDER_CONFIGS = [
        COTEConfig(
            name="bd",
            data_url=_COTE_URLs["bd"],
            data_dir="COTE-BD",
            version=datasets.Version("1.0.0", ""),
            description="COTE-BD crawled on baidu.",
        ),
        COTEConfig(
            name="mfw",
            data_url=_COTE_URLs["mfw"],
            data_dir="COTE-MFW",
            version=datasets.Version("1.0.0", ""),
            description="COTE-MFW crawled on Mafengwo.",
        ),
        COTEConfig(
            name="dp",
            data_url=_COTE_URLs["dp"],
            data_dir="COTE-DP",
            version=datasets.Version("1.0.0", ""),
            description="COTE-DP crawled on Dianping.",
        ),
    ]

    def _info(self):
        features = {
            "id": datasets.Value("int32"),
            "text_a": datasets.Value("string"),
            "label": datasets.Value("string"),
        }

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
            homepage="https://www.luge.ai/#/luge/dataDetail?id=19",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_dir = dl_manager.download_and_extract(self.config.data_url)
        data_dir = os.path.join(downloaded_dir, self.config.data_dir)

        train_split = datasets.SplitGenerator(
            name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(data_dir, "train.tsv"), "split": "train"}
        )
        test_split = datasets.SplitGenerator(
            name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(data_dir, "test.tsv"), "split": "test"}
        )

        return [train_split, test_split]

    def _generate_examples(self, filepath, split):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)

        with open(filepath, encoding="utf8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)

            for idx, row in enumerate(reader):
                example = {}
                example["id"] = idx
                example["text_a"] = row["text_a"]

                if split == "train":
                    example["label"] = row["label"]
                else:
                    example["label"] = ""

                # Filter out corrupted rows.
                for value in example.values():
                    if value is None:
                        break
                else:
                    yield idx, example
