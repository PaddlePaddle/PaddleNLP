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

    If 'ori' is passed to `load_dataset`, the returned dataset contain original
    data, whose labels might be '-' additionally, which means five signers did
    not reach a consensus. If 'clean' is chosen, examples contain label '-'
    would be filtered out. 'clean' is recommended choice for training.

    More information please refer to `https://github.com/cluebenchmark/OCNLI`
    '''
    BUILDER_CONFIGS = {
        'ori': {
            'url': "https://paddlenlp.bj.bcebos.com/datasets/ocnli.zip",
            'md5': "acb426f6f3345076c6ce79239e7bc307",
            'splits': {
                'train': [
                    os.path.join('train.50k.json'),
                    'd38ec492ef086a894211590a18ab7596',
                ],
                'dev': [
                    os.path.join('dev.json'),
                    '3481b456bee57a3c9ded500fcff6834c',
                ],
                'test': [
                    os.path.join('test.json'),
                    '680ff24e6b3419ff8823859bc17936aa',
                ]
            },
            'labels': ["entailment", "contradiction", "neutral", "-"]
        },
        'clean': {
            'url': "https://paddlenlp.bj.bcebos.com/datasets/ocnli.zip",
            'md5': "acb426f6f3345076c6ce79239e7bc307",
            'splits': {
                'train': [
                    os.path.join('train.50k.json'),
                    'd38ec492ef086a894211590a18ab7596',
                ],
                'dev': [
                    os.path.join('dev.json'),
                    '3481b456bee57a3c9ded500fcff6834c',
                ],
                'test': [
                    os.path.join('test.json'),
                    '680ff24e6b3419ff8823859bc17936aa',
                ]
            },
            'labels': ["entailment", "contradiction", "neutral"]
        }
    }

    def _get_data(self, mode, **kwargs):
        builder_config = self.BUILDER_CONFIGS[self.name]
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = builder_config['splits'][mode]
        fullname = os.path.join(default_root, filename)

        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(builder_config['url'], default_root,
                              builder_config['md5'])
        return fullname

    def _read(self, filename, split):
        if self.name == 'clean' and split != 'test':
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    example_dict = json.loads(line.rstrip())
                    if example_dict['label'] == '-':
                        continue
                    yield example_dict
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    yield json.loads(line.rstrip())

    def get_labels(self):
        """
        Returns labels of the OCNLI object.
        """
        return self.BUILDER_CONFIGS[self.name]['labels']
