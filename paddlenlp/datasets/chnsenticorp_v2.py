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
import os

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url

from ..utils.env import DATA_HOME
from .dataset import DatasetBuilder

__all__ = ["ChnSentiCorpV2"]


class ChnSentiCorpV2(DatasetBuilder):
    """
    ChnSentiCorp (by Tan Songbo at ICT of Chinese Academy of Sciences, and for
    opinion mining)

    """

    URL = "https://paddlenlp.bj.bcebos.com/datasets/data-chnsenticorp.tar.gz"
    MD5 = "e336e76d7be4ecd5479083d5b8f771e4"
    META_INFO = collections.namedtuple("META_INFO", ("file", "md5"))
    SPLITS = {
        "train": META_INFO(os.path.join("chnsenticorp", "train", "part.0"), "3fac2659547f1ddf90d223b8ed31f22f"),
        "dev": META_INFO(os.path.join("chnsenticorp", "dev", "part.0"), "a3a853bfb3af4a592fc4df24b56c88a7"),
        "test": META_INFO(os.path.join("chnsenticorp", "test", "part.0"), "6bfc8f35f523d2fdf12648d9d02778ff"),
    }

    def _get_data(self, mode, **kwargs):
        """Downloads dataset."""
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and not md5file(fullname) == data_hash):
            get_path_from_url(self.URL, default_root, self.MD5)

        return fullname

    def _read(self, filename, split):
        """Reads data."""
        with open(filename, "r", encoding="utf-8") as f:
            head = True
            for line in f:
                data = line.strip().split("\t")
                if not head:
                    head = data
                else:
                    if split == "train":
                        text, label = data
                        yield {"text": text, "label": label}
                    elif split == "dev":
                        text, label = data
                        yield {"text": text, "label": label}
                    elif split == "test":
                        text, label = data
                        yield {"text": text, "label": label}

    def get_labels(self):
        """
        Return labels of the ChnSentiCorp object.
        """
        return ["0", "1"]
