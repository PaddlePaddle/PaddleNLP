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
"""SE-ABSA16: SemEval-2016 Task 5: Aspect Based Sentiment Analysis."""

import csv
import os
import textwrap
import numpy as np

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@inproceedings{pontiki2016semeval,
  title={Semeval-2016 task 5: Aspect based sentiment analysis},
  author={Pontiki, Maria and Galanis, Dimitrios and Papageorgiou, Haris and Androutsopoulos, Ion and Manandhar, Suresh and Al-Smadi, Mohammad and Al-Ayyoub, Mahmoud and Zhao, Yanyan and Qin, Bing and De Clercq, Orph{\'e}e and others},
  booktitle={International workshop on semantic evaluation},
  pages={19--30},
  year={2016}
}
"""

_DESCRIPTION = """\
SE-ABSA16, a dataset for aspect based sentiment analysis, which aims to perform fine-grained sentiment classification for aspect in text. The dataset contains both positive and negative categories. It covers the data of mobile phone and camera.
More information refer to https://www.luge.ai/#/luge/dataDetail?id=18.
"""

_SEABSA16_URLs = {
    # pylint: disable=line-too-long
    "came": "https://paddlenlp.bj.bcebos.com/datasets/SE-ABSA16_CAME.zip",
    "phns": "https://paddlenlp.bj.bcebos.com/datasets/SE-ABSA16_PHNS.zip",
    # pylint: enable=line-too-long
}


class SEABSA16Config(datasets.BuilderConfig):
    """BuilderConfig for SEABSA16."""

    def __init__(self, data_url=None, data_dir=None, **kwargs):
        """BuilderConfig for SEABSA16.

        Args:
          data_url: `string`, url to download the zip file.
          data_dir: `string`, the path to the folder containing the tsv files in the downloaded zip.
          **kwargs: keyword arguments forwarded to super.
        """
        super(SEABSA16Config, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir


class SEABSA16(datasets.GeneratorBasedBuilder):
    """SE-ABSA16: SemEval-2016 Task 5: Aspect Based Sentiment Analysis."""

    BUILDER_CONFIGS = [
        SEABSA16Config(
            name="came",
            data_url=_SEABSA16_URLs["came"],
            data_dir="SE-ABSA16_CAME",
            version=datasets.Version("1.0.0", ""),
            description="SE-ABSA16-CAME data about camera.",
        ),
        SEABSA16Config(
            name="phns",
            data_url=_SEABSA16_URLs["phns"],
            data_dir="SE-ABSA16_PHNS",
            version=datasets.Version("1.0.0", ""),
            description="SE-ABSA16-PHNS data about phone.",
        ),
    ]

    def _info(self):
        features = {
            "id": datasets.Value("int32"),
            "text_a": datasets.Value("string"),
            "text_b": datasets.Value("string"),
            "label": datasets.Value("int32"),
        }

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
            homepage="https://www.luge.ai/#/luge/dataDetail?id=18",
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
                example["text_b"] = row["text_b"]

                if split == "train":
                    example["label"] = int(row["label"])
                else:
                    example["label"] = -1

                # Filter out corrupted rows.
                for value in example.values():
                    if value is None:
                        break
                else:
                    yield idx, example
