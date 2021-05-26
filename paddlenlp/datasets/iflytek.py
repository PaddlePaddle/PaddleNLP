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


class IFLYTEK(DatasetBuilder):
    '''
    IFLYTEK contains 17,332 app descriptions. The task is to assign each
    description into one of 119 categories, such as food, car rental,
    education, etc. 

    More information please refer to `https://github.com/CLUEbenchmark/CLUE`
    '''
    URL = "https://paddlenlp.bj.bcebos.com/datasets/iflytek.zip"
    MD5 = "19e4b19947db126f69aae18db0da2b87"
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('train.json'), 'fc9a21700c32ee3efee3fc283e9ac560'),
        'dev': META_INFO(
            os.path.join('dev.json'), '79b7d95bddeb11cd54198fd077992704'),
        'test': META_INFO(
            os.path.join('test.json'), 'ea764519ddb4369767d07664afde3325'),
        'labels': META_INFO(
            os.path.join('labels.json'), '7f9e794688ffb37fbd42b58325579fdf')
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
        Returns labels of the TNEWS object.
        """
        labels = [str(i) for i in range(119)]
        return labels
