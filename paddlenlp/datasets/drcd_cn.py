# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = ['DRCD_CN']


class DRCD_CN(DatasetBuilder):
    '''
    Delta Reading Comprehension Dataset is an open domain traditional Chinese
    machine reading comprehension (MRC) dataset. The dataset contains 10,014
    paragraphs from 2,108 Wikipedia articles and 30,000+ questions generated
    by annotators.

    This dataset translate origin Traditional Chinese to Simplified Chinese.
    '''

    URL = "https://bj.bcebos.com/paddlenlp/datasets/drcd_cn.tar.gz"
    MD5 = "8ceed5076c4f59d7a3666b13851e41fa"
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train':
        META_INFO(os.path.join('drcd_cn', 'train.json'),
                  '5a51ee5a106e16965c85fce364d316d7'),
        'dev':
        META_INFO(os.path.join('drcd_cn', 'dev.json'),
                  'f352b17cddeed69877ff94d4321817ce'),
        'test':
        META_INFO(os.path.join('drcd_cn', 'test.json'),
                  'e674a667033c4e8c9ae6d05d95073d02')
    }

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(self.URL, default_root)

        return fullname

    def _read(self, filename, *args):
        with open(filename, "r", encoding="utf8") as f:
            input_data = json.load(f)["data"]
        for entry in input_data:
            title = entry.get("title", "").strip()
            for paragraph in entry["paragraphs"]:
                context = paragraph["context"].strip()
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question = qa["question"].strip()
                    answer_starts = [
                        answer["answer_start"]
                        for answer in qa.get("answers", [])
                    ]
                    answers = [
                        answer["text"].strip()
                        for answer in qa.get("answers", [])
                    ]

                    yield {
                        'id': qas_id,
                        'title': title,
                        'context': context,
                        'question': question,
                        'answers': answers,
                        'answer_starts': answer_starts
                    }
