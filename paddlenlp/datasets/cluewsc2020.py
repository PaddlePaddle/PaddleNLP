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


class CLUEWSC2020(DatasetBuilder):
    '''
    The Chinese Winograd Schema Challenge dataset is an anaphora/coreference
    resolution task where the model is asked to decide whether a pronoun and a
    noun (phrase) in a sentence co-refer (binary classification), built
    following similar datasets in English.

    More information please refer to `https://github.com/CLUEbenchmark/CLUE`
    '''
    URL = "https://paddlenlp.bj.bcebos.com/datasets/cluewsc2020.tar.gz"
    MD5 = "17abe1be3f7dd3bad5f114ba4c40ee9b"
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('train.json'), 'afd235dcf8cdb89ee1a21d0a4823eecc'),
        'dev': META_INFO(
            os.path.join('dev.json'), 'bad8cd6fa0916fc37ac96b8ce316714a'),
        'test': META_INFO(
            os.path.join('test.json'), '0e9e8ffd8ee90ddf1f58d6dc2e02de7b')
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
        Returns labels of the CLUEWSC2020 object.
        """
        return ["true", "false"]
