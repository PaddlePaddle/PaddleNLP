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


class AFQMC(DatasetBuilder):
    '''
    AFQMC: The Ant Financial Question Matching Corpus3 comes from Ant Technology
    Exploration Conference (ATEC) Developer competition. It is a binary
    classification task that aims to predict whether two sentences are
    semantically similar.

    More information please refer to `https://github.com/CLUEbenchmark/CLUE`
    '''
    URL = "https://paddlenlp.bj.bcebos.com/datasets/afqmc.zip"
    MD5 = "3377b559bb4e61d03a35282550902ca0"
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('train.json'), '319cf775353af9473140abca4052b89a'),
        'dev': META_INFO(
            os.path.join('dev.json'), '307154b59cb6c3e68a0f39c310bbd364'),
        'test': META_INFO(
            os.path.join('test.json'), '94b925f23a9615dd08199c4013f761f4')
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
        Returns labels of the AFQMC object.
        """
        return ["0", "1"]
