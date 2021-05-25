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


class TNEWS(DatasetBuilder):
    '''
    TouTiao Text Classification for News Titles2 consists of Chinese news
    published by TouTiao before May 2018, with a total of 73,360 titles. Each
    title is labeled with one of 15 news categories (finance, technology,
    sports, etc.) and the task is to predict which category the title belongs
    to.

    More information please refer to `https://github.com/CLUEbenchmark/CLUE`
    '''
    URL = "https://paddlenlp.bj.bcebos.com/datasets/tnews.tar.gz"
    MD5 = "587171233c8e8db00a3dc9bae5d2b47d"
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('train.json'), '25c021725309a3330736380a230850fd'),
        'dev': META_INFO(
            os.path.join('dev.json'), 'f0660a3339a32e764075c801b42ece3c'),
        'test': META_INFO(
            os.path.join('test.json'), '2d1557c7548c72d5a84c47bbbd3a4e85'),
        'labels': META_INFO(
            os.path.join('labels.json'), 'a1a7595e596b202556dedd2a20617769')
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
        labels = [str(i) for i in range(100, 117)]
        return labels
