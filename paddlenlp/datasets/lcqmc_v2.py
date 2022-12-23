# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import json
import os

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ["LCQMC_V2"]


class LCQMC_V2(DatasetBuilder):
    """
    LCQMC:A Large-scale Chinese Question Matching Corpus
    More information please refer to `https://www.aclweb.org/anthology/C18-1166/`

    """

    URL = "https://bj.bcebos.com/paddlenlp/datasets/lcqmc_v2.tar.gz"
    MD5 = "e44825d8e6d5117bc04caf3982cf934f"
    META_INFO = collections.namedtuple("META_INFO", ("file", "md5"))
    SPLITS = {
        "train": META_INFO(os.path.join("lcqmc", "train.tsv"), "2193c022439b038ac12c0ae918b211a1"),
        "dev": META_INFO(os.path.join("lcqmc", "dev.tsv"), "c5dcba253cb4105d914964fd8b3c0e94"),
        "test": META_INFO(os.path.join("lcqmc", "test.tsv"), "8f4b71e15e67696cc9e112a459ec42bd"),
    }

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and not md5file(fullname) == data_hash):
            get_path_from_url(self.URL, default_root, self.MD5)

        return fullname

    def _read(self, filename):
        """Reads data."""
        with open(filename, "r", encoding="utf-8") as f:
            head = True
            for line in f:
                data = line.strip().split("\t")
                if head:
                    head = False
                else:
                    if len(data) == 3:
                        query, title, label = data
                        yield {"query": query, "title": title, "label": label}
                    elif len(data) == 2:
                        query, title = data
                        yield {"query": query, "title": title, "label": ""}
                    else:
                        continue

    def get_labels(self):
        """
        Return labels of the LCQMC object.
        """
        return ["0", "1"]
