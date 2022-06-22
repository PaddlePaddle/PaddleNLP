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

# Copyright (c) 2017 Kamran Kowsari
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this dataset and associated documentation files (the "Dataset"), to deal
# in the dataset without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Dataset, and to permit persons to whom the dataset is
# furnished to do so, subject to the following conditions:

import collections
import os
import warnings

from paddle.io import Dataset
from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from paddlenlp.datasets import DatasetBuilder

__all__ = ['WOS']


class WOS(DatasetBuilder):
    """
    Web of Science(WOS) dataset contains abstracts of published papers from Web of Science.
    More information please refer to 'https://data.mendeley.com/datasets/9rw3vkcfy4/2'.
    """
    lazy = False
    URL = "https://bj.bcebos.com/paddlenlp/datasets/wos.tar.gz"
    MD5 = '81190b3d8aa2e94b40d58adf296404ee'
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train':
        META_INFO(os.path.join('wos', 'train.tsv'),
                  '91d833883d0ecf7395a4fd0373a4b395'),
        'dev':
        META_INFO(os.path.join('wos', 'dev.tsv'),
                  'afec59209a140057a0a204e8e99c14ac'),
        'test':
        META_INFO(os.path.join('wos', 'test.tsv'),
                  '18f4eaec8b94d5b49193d4d9e58d5528')
    }

    def _get_data(self, mode, **kwargs):
        ''' Check and download Dataset '''
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):

            get_path_from_url(self.URL, default_root, self.MD5)

        return fullname

    def _read(self, filename, *args):

        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line_stripped = line.strip().split('\t')
                sentence, level1, level2 = line_stripped

                yield {
                    "sentence": sentence,
                    "level 1": level1.split(','),
                    "level 2": level2.split(',')
                }
