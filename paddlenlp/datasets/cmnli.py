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


class CMNLI(DatasetBuilder):
    '''
    CMNLI: Chinese Multi-Genre NLI.

    If 'ori' is passed to `load_dataset`, the returned develop dataset contain
    original data, whose labels might be '-' additionally, which means five
    signers did not reach a consensus. If 'clean' is chosen, examples contain
    label '-' would be filtered out. 'clean' is recommended choice for training.

    More information please refer to `https://github.com/CLUEbenchmark/CLUE`
    '''
    BUILDER_CONFIGS = {
        'ori': {
            'url': "https://paddlenlp.bj.bcebos.com/datasets/cmnli.zip",
            'md5': "e0e8caefd9b3491220c18b466233f2ff",
            'splits': {
                'train': [
                    os.path.join('cmnli_public', 'train.json'),
                    '7d02308650cd2a0e183bf599ca9bb263',
                ],
                'dev': [
                    os.path.join('cmnli_public', 'dev.json'),
                    '0b16a50a297a9afb1ce5385ee4dd3d9c',
                ],
                'test': [
                    os.path.join('cmnli_public', 'test.json'),
                    '804cb0bb67266983d59d1c855e6b03b0',
                ]
            },
            'labels': ["entailment", "contradiction", "neutral", "-"]
        },
        'clean': {
            'url': "https://paddlenlp.bj.bcebos.com/datasets/cmnli.zip",
            'md5': "e0e8caefd9b3491220c18b466233f2ff",
            'splits': {
                'train': [
                    os.path.join('cmnli_public', 'train.json'),
                    '7d02308650cd2a0e183bf599ca9bb263',
                ],
                'dev': [
                    os.path.join('cmnli_public', 'dev.json'),
                    '0b16a50a297a9afb1ce5385ee4dd3d9c',
                ],
                'test': [
                    os.path.join('cmnli_public', 'test.json'),
                    '804cb0bb67266983d59d1c855e6b03b0',
                ]
            },
            'labels': ["entailment", "contradiction", "neutral"]
        },
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
        if self.name == 'clean' and split == 'dev':
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
