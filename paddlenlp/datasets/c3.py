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
__all__ = ['C3']


class C3(DatasetBuilder):
    '''
    The first free-form multiple-Choice Chinese machine reading Comprehension dataset.
    See more details on 
    '''
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5', 'URL'))
    SPLITS = {
        'train': [
            META_INFO(
                os.path.join('c3-d-train.json'),
                '291b07679bef785aa66bb5343f1b49b2',
                'https://paddlenlp.bj.bcebos.com/datasets/c3/c3-d-train.json'),
            META_INFO(
                os.path.join('c3-m-train.json'),
                'db321e631eb3e6f508e438992652618f',
                'https://paddlenlp.bj.bcebos.com/datasets/c3/c3-m-train.json'),
        ],
        'dev': [
            META_INFO(
                os.path.join('c3-d-dev.json'),
                '446e75358789d3fbe8730089cadf5fb0',
                'https://paddlenlp.bj.bcebos.com/datasets/c3/c3-d-dev.json'),
            META_INFO(
                os.path.join('c3-m-dev.json'),
                'beb2f2e08c18cd8e9429c6a55de6b8db',
                'https://paddlenlp.bj.bcebos.com/datasets/c3/c3-m-dev.json'),
        ],
        'test': [
            META_INFO(
                os.path.join('c3-d-test.json'),
                '002561f15f4942328761c50c90ced36c',
                'https://paddlenlp.bj.bcebos.com/datasets/c3/c3-d-test.json'),
            META_INFO(
                os.path.join('c3-m-test.json'),
                'f5f14c517926d22047b7bfd369dab724',
                'https://paddlenlp.bj.bcebos.com/datasets/c3/c3-m-test.json'),
        ],
    }

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__, mode)
        meta_info_list = self.SPLITS[mode]

        for meta_info in meta_info_list:
            filename, data_hash, URL = meta_info
            fullname = os.path.join(default_root, filename)
            if not os.path.exists(fullname) or (
                    data_hash and not md5file(fullname) == data_hash):
                get_path_from_url(URL, default_root)

        return default_root

    def _read(self, data_dir, *args):
        data_files = os.listdir(data_dir)
        for files in data_files:
            fullname = os.path.join(data_dir, files)
            with open(fullname, "r", encoding='utf8') as fr:
                samples = json.load(fr)
                for sample in samples:
                    context = sample[0]
                    qas = sample[1]
                    for qa in qas:
                        question = qa['question']
                        choice = qa['choice']
                        answer = qa['answer']
                        yield {
                            'context': context,
                            'question': question,
                            'choice': choice,
                            'answer': answer,
                        }

    def get_labels(self):
        """
        Return labels of the C3 object.
        """
        return ["0", "1", "2", "3"]
