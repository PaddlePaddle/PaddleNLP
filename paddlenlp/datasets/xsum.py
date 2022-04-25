# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = ['xsum']


class xsum(DatasetBuilder):

    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5', 'URL'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('train.json'), '4e06fd1cfd5e7f0380499df8cbe17237',
            'https://bj.bcebos.com/paddlenlp/datasets/xsum/train.json'),
        'dev': META_INFO(
            os.path.join('dev.json'), '9c39d49d25d5296bdc537409208ddc85',
            'https://bj.bcebos.com/paddlenlp/datasets/xsum/dev.json'),
        'test': META_INFO(
            os.path.join('test.json'), '9c39d49d25d5296bdc537409208ddc85',
            'https://bj.bcebos.com/paddlenlp/datasets/xsum/test.json')
    }

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash, URL = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(URL, default_root)
        return fullname

    def _read(self, filename, split):
        with open(filename, encoding="utf-8") as f:
            xsums = json.load(f)
            for data in xsums:
                yield {"document": data["document"], "label": data["summary"],"id": data["id"]}
