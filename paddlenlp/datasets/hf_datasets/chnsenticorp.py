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
"""ChnSentiCorp: Chinese Corpus for sentence-level sentiment classification."""

import csv
import os
import textwrap
import numpy as np

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@article{tan2008empirical,
  title={An empirical study of sentiment analysis for chinese documents},
  author={Tan, Songbo and Zhang, Jin},
  journal={Expert Systems with applications},
  volume={34},
  number={4},
  pages={2622--2629},
  year={2008},
  publisher={Elsevier}
}
"""

_DESCRIPTION = """\
ChnSentiCorp: A classic sentence-level sentiment classification dataset, which includes hotel, laptop and data-related online review data, including positive and negative categories.
More information refer to https://www.luge.ai/#/luge/dataDetail?id=25.
"""

_URL = "https://bj.bcebos.com/paddlenlp/datasets/ChnSentiCorp.zip"


class ChnSentiCorpConfig(datasets.BuilderConfig):
    """BuilderConfig for ChnSentiCorp."""

    def __init__(self, **kwargs):
        """BuilderConfig for ChnSentiCorp.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ChnSentiCorpConfig, self).__init__(**kwargs)


class ChnSentiCorp(datasets.GeneratorBasedBuilder):
    """ChnSentiCorp: Chinese Corpus for sentence-level sentiment classification."""

    BUILDER_CONFIGS = [
        ChnSentiCorpConfig(
            name="chnsenticorp",
            version=datasets.Version("1.0.0", ""),
            description="COTE-BD crawled on baidu.",
        )
    ]

    def _info(self):
        features = {"id": datasets.Value("int32"), "text": datasets.Value("string"), "label": datasets.Value("int32")}

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
            homepage="https://www.luge.ai/#/luge/dataDetail?id=25",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(downloaded_dir, "ChnSentiCorp")

        train_split = datasets.SplitGenerator(
            name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(data_dir, "train.tsv"), "split": "train"}
        )

        dev_split = datasets.SplitGenerator(
            name=datasets.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(data_dir, "dev.tsv"), "split": "dev"}
        )

        test_split = datasets.SplitGenerator(
            name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(data_dir, "test.tsv"), "split": "test"}
        )

        return [train_split, dev_split, test_split]

    def _generate_examples(self, filepath, split):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)

        with open(filepath, encoding="utf8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)

            for idx, row in enumerate(reader):
                example = {}
                example["id"] = idx
                example["text"] = row["text_a"]

                if split != "test":
                    example["label"] = int(row["label"])
                else:
                    example["label"] = -1

                # Filter out corrupted rows.
                for value in example.values():
                    if value is None:
                        break
                else:
                    yield idx, example
