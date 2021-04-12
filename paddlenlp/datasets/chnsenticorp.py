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

__all__ = ['ChnSentiCorp']


class ChnSentiCorp(DatasetBuilder):
    """
    ChnSentiCorp (by Tan Songbo at ICT of Chinese Academy of Sciences, and for
    opinion mining)

    """

    URL = "https://bj.bcebos.com/paddlehub-dataset/chnsenticorp.tar.gz"
    MD5 = "fbb3217aeac76a2840d2d5cd19688b07"
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('chnsenticorp', 'train.tsv'),
            '689360c4a4a9ce8d8719ed500ae80907'),
        'dev': META_INFO(
            os.path.join('chnsenticorp', 'dev.tsv'),
            '05e4b02561c2a327833e05bbe8156cec'),
        'test': META_INFO(
            os.path.join('chnsenticorp', 'test.tsv'),
            '917dfc6fbce596bb01a91abaa6c86f9e'),
    }

    def _get_data(self, mode, **kwargs):
        """Downloads dataset."""
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(self.URL, default_root, self.MD5)

        return fullname

    def _read(self, filename):
        """Reads data."""
        with open(filename, 'r', encoding='utf-8') as f:
            head = None
            for line in f:
                data = line.strip().split("\t")
                if not head:
                    head = data
                else:
                    label, text = data
                    yield {"text": text, "label": label}

    def get_labels(self):
        """
        Return labels of the ChnSentiCorp object.
        """
        return ["0", "1"]
