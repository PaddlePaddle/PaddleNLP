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


class OCNLI(DatasetBuilder):
    '''
    Original Chinese Natural Language Inference is collected closely following
    procedures of MNLI. OCNLI is composed of 56k inference pairs from five
    genres: news, government, fiction, TV transcripts and Telephone transcripts,
    where the premises are collected from Chinese sources, and universities
    students in language majors are hired to write the hypotheses.

    More information please refer to `https://github.com/cluebenchmark/OCNLI`
    '''
    URL = "https://paddlenlp.bj.bcebos.com/datasets/ocnli.tar.gz"
    MD5 = "acb426f6f3345076c6ce79239e7bc307"
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('train.50k.json'), 'd38ec492ef086a894211590a18ab7596'),
        'dev': META_INFO(
            os.path.join('dev.json'), '3481b456bee57a3c9ded500fcff6834c'),
        'test': META_INFO(
            os.path.join('test.json'), '680ff24e6b3419ff8823859bc17936aa')
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
        Returns labels of the OCNLI object.
        """
        return ["entailment", "contradiction", "neutral"]
