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


class CSL(DatasetBuilder):
    '''
    Chinese Scientific Literature dataset contains Chinese paper abstracts and
    their keywords from core journals of China, covering multiple fields of
    natural sciences and social sciences.

    More information please refer to `https://github.com/P01son6415/CSL`
    '''
    URL = "https://paddlenlp.bj.bcebos.com/datasets/csl.zip"
    MD5 = "394a2ccbf6ddd7e331be4d5d7798f0f6"
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('train.json'), 'e927948b4e0eb4992fe9f45a77446bf5'),
        'dev': META_INFO(
            os.path.join('dev.json'), '6c2ab8dd3b4785829ead94b05a1cb957'),
        'test': META_INFO(
            os.path.join('test.json'), 'ebfb89575355f00dcd9b18f8353547cd')
    }

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(self.URL, default_root, self.MD5)

        return fullname

    def _read(self, filename, split):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line.rstrip())

    def get_labels(self):
        """
        Returns labels of the CSL object.
        """
        return ["0", "1"]
