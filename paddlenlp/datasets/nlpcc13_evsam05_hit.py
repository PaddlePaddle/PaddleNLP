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

__all__ = ['NLPCC13EVSAM05HIT']


class NLPCC13EVSAM05HIT(DatasetBuilder):

    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5', 'URL'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('evsam05', '╥└┤µ╖╓╬÷╤╡┴╖╩²╛▌', 'HIT', 'train.conll'),
            'c7779f981203b4ecbe5b04c65aaaffce',
            'http://tcci.ccf.org.cn/conference/2013/dldoc/evsam05.zip',
        ),
        'dev': META_INFO(
            os.path.join('evsam05', '╥└┤µ╖╓╬÷╤╡┴╖╩²╛▌', 'HIT', 'dev.conll'),
            '59c2de72c7be39977f766e8290336dac',
            'http://tcci.ccf.org.cn/conference/2013/dldoc/evsam05.zip',
        ),
        'test': META_INFO(
            os.path.join('▓Γ╩╘┤≡░╕', 'HIT', 'golden.conll'),
            '91ae33dc21adace18788885298a3155a',
            'http://tcci.ccf.org.cn/conference/2013/dldoc/evans05.zip',
        ),
    }

    def _get_data(self, mode, **kwargs):
        """Downloads dataset."""
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash, URL = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(URL, default_root)

        return fullname

    def _read(self, filename, split):
        start = 0
        with open(filename, 'r', encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                if not line.startswith(" "):
                    if not line.startswith('#') and (len(line) == 1 or line.split()[0].isdigit()):
                        lines.append(line.strip())
                else:
                    lines.append("")

        for i, line in enumerate(lines):
            if not line:
                values = list(zip(*[j.split('\t') for j in lines[start:i]]))
                _, FORM, _, CPOS, _, _, HEAD, DEPREL, _, _ = values
                if values:
                    yield {"FORM": FORM, "CPOS": CPOS, "HEAD": HEAD, "DEPREL": DEPREL}
                start = i + 1

        