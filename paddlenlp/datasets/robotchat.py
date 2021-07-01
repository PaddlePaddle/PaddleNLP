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

__all__ = ['RobotChat']


class RobotChat(DatasetBuilder):
    """
    RobotChat (an open source robot chat dataset)

    """

    URL = "https://paddlenlp.bj.bcebos.com/datasets/RobotChat.tar.gz"
    MD5 = "9d0122c1d091d0eba8c7561f76e217d5"
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('RobotChat', 'train.tsv'),
            '2dbf2e02b87b91fa18a897bdb8a37410'),
        'dev': META_INFO(
            os.path.join('RobotChat', 'dev.tsv'),
            'a06e434f64749ebf372a4f14b40ab61e'),
        'test': META_INFO(
            os.path.join('RobotChat', 'test.tsv'),
            'c600d3d22f772aa684d809326d9a7135'),
    }

    def _get_data(self, mode, **kwargs):
        """Downloads dataset."""
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not (fullname) == data_hash):
            get_path_from_url(self.URL, default_root, self.MD5)

        return fullname

    def _read(self, filename, split):
        """Reads data."""
        with open(filename, 'r', encoding='utf-8') as f:
            head = None
            for line in f:
                data = line.strip().split("\t")
                if not head:
                    head = data
                else:
                    if split == 'train' or split == 'dev':   
                        label, text = data
                        yield {"text": text, "label": label, "qid": ''}
                    elif split == 'test':
                        label, text = data
                        yield {"text": text, "label": '', "qid": ''}

    def get_labels(self):
        """
        Return labels of the ChnSentiCorp object.
        """
        return ["0", "1", "2"]
