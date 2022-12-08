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

import datasets

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """
LanguagePairDataset used for machine translation between any pair of languages. """

_URL = "https://bj.bcebos.com/paddlenlp/datasets/WMT14.en-de.tar.gz"


class LanguagePairConfig(datasets.BuilderConfig):
    """BuilderConfig for a general LanguagePairDataset."""

    def __init__(self, **kwargs):
        """BuilderConfig for LanguagePairDataset.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(LanguagePairConfig, self).__init__(**kwargs)


class LanguagePairDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        LanguagePairConfig(
            name="LanguagePair",
            version=datasets.Version("1.0.0", ""),
            description=_DESCRIPTION,
        ),
    ]

    def _info(self):
        logger.warning(
            "LanguagePairDataset is an experimental API which we will continue to optimize and may be changed."
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "source": datasets.Value("string"),
                    "target": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        is_downloaded = False

        # Train files.
        if hasattr(self.config, "data_files") and "train" in self.config.data_files:
            train_split = datasets.SplitGenerator(
                name="train",
                gen_kwargs={
                    "source_filepath": os.path.abspath(self.config.data_files["train"][0]),
                    "target_filepath": os.path.abspath(self.config.data_files["train"][1]),
                },
            )

        else:
            if not is_downloaded:
                dl_dir = dl_manager.download_and_extract(_URL)
                is_downloaded = True
            train_split = datasets.SplitGenerator(
                name="train",
                gen_kwargs={
                    "source_filepath": os.path.join(
                        dl_dir, "WMT14.en-de", "wmt14_ende_data_bpe", "train.tok.clean.bpe.33708.en"
                    ),
                    "target_filepath": os.path.join(
                        dl_dir, "WMT14.en-de", "wmt14_ende_data_bpe", "train.tok.clean.bpe.33708.de"
                    ),
                },
            )

        # Dev files.
        if hasattr(self.config, "data_files") and "dev" in self.config.data_files:
            dev_split = datasets.SplitGenerator(
                name="dev",
                gen_kwargs={
                    "source_filepath": os.path.abspath(self.config.data_files["dev"][0]),
                    "target_filepath": os.path.abspath(self.config.data_files["dev"][1]),
                },
            )

        else:
            if not is_downloaded:
                dl_dir = dl_manager.download_and_extract(_URL)
                is_downloaded = True
            dev_split = datasets.SplitGenerator(
                name="dev",
                gen_kwargs={
                    "source_filepath": os.path.join(
                        dl_dir, "WMT14.en-de", "wmt14_ende_data_bpe", "newstest2013.tok.bpe.33708.en"
                    ),
                    "target_filepath": os.path.join(
                        dl_dir, "WMT14.en-de", "wmt14_ende_data_bpe", "newstest2013.tok.bpe.33708.de"
                    ),
                },
            )

        # Test files.
        if hasattr(self.config, "data_files") and "test" in self.config.data_files:
            # test may not contain target languages.
            if isinstance(self.config.data_files["test"], str):
                self.config.data_files["test"] = [self.config.data_files["test"], None]
            elif (
                isinstance(self.config.data_files["test"], (list, tuple)) and len(self.config.data_files["test"]) == 1
            ):
                self.config.data_files["test"].append(None)

            test_split = datasets.SplitGenerator(
                name="test",
                gen_kwargs={
                    "source_filepath": os.path.abspath(self.config.data_files["test"][0]),
                    "target_filepath": os.path.abspath(self.config.data_files["test"][1]),
                },
            )

        else:
            if not is_downloaded:
                dl_dir = dl_manager.download_and_extract(_URL)
                is_downloaded = True
            test_split = datasets.SplitGenerator(
                name="test",
                gen_kwargs={
                    "source_filepath": os.path.join(
                        dl_dir, "WMT14.en-de", "wmt14_ende_data_bpe", "newstest2014.tok.bpe.33708.en"
                    ),
                    "target_filepath": os.path.join(
                        dl_dir, "WMT14.en-de", "wmt14_ende_data_bpe", "newstest2014.tok.bpe.33708.de"
                    ),
                },
            )

        return [train_split, dev_split, test_split]

    def _generate_examples(self, source_filepath, target_filepath):
        """This function returns the examples in the raw (text) form."""

        logger.info("generating examples from = source: {} & target: {}".format(source_filepath, target_filepath))
        key = 0

        with open(source_filepath, "r", encoding="utf-8") as src_fin:
            if target_filepath is not None:
                with open(target_filepath, "r", encoding="utf-8") as tgt_fin:
                    src_seq = src_fin.readlines()
                    tgt_seq = tgt_fin.readlines()

                    for i, src in enumerate(src_seq):
                        source = src.strip()
                        target = tgt_seq[i].strip()

                        yield key, {
                            "id": str(key),
                            "source": source,
                            "target": target,
                        }
                        key += 1
            else:
                src_seq = src_fin.readlines()
                for i, src in enumerate(src_seq):
                    source = src.strip()

                    yield key, {
                        "id": str(key),
                        "source": source,
                        # None is not allowed.
                        "target": "",
                    }
                    key += 1
