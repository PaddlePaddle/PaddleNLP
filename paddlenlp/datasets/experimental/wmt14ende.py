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
import os
import warnings

from paddle.io import Dataset
from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['WMT14ende']


class WMT14ende(DatasetBuilder):
    URL = "https://paddlenlp.bj.bcebos.com/datasets/WMT14.en-de.tar.gz"
    MD5 = None
    META_INFO = collections.namedtuple('META_INFO', ('src_file', 'tgt_file',
                                                     'src_md5', 'tgt_md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                         "train.tok.clean.bpe.33708.en"),
            os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                         "train.tok.clean.bpe.33708.de"),
            "c7c0b77e672fc69f20be182ae37ff62c", None),
        'dev': META_INFO(
            os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                         "newstest2013.tok.bpe.33708.en"),
            os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                         "newstest2013.tok.bpe.33708.de"),
            "aa4228a4bedb6c45d67525fbfbcee75e",
            "9b1eeaff43a6d5e78a381a9b03170501"),
        'test': META_INFO(
            os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                         "newstest2014.tok.bpe.33708.en"),
            os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                         "newstest2014.tok.bpe.33708.de"),
            "c9403eacf623c6e2d9e5a1155bdff0b5",
            "0058855b55e37c4acfcb8cffecba1050"),
        'dev-eval': META_INFO(
            os.path.join("WMT14.en-de", "wmt14_ende_data",
                         "newstest2013.tok.en"),
            os.path.join("WMT14.en-de", "wmt14_ende_data",
                         "newstest2013.tok.de"),
            "d74712eb35578aec022265c439831b0e",
            "6ff76ced35b70e63a61ecec77a1c418f"),
        'test-eval': META_INFO(
            os.path.join("WMT14.en-de", "wmt14_ende_data",
                         "newstest2014.tok.en"),
            os.path.join("WMT14.en-de", "wmt14_ende_data",
                         "newstest2014.tok.de"),
            "8cce2028e4ca3d4cc039dfd33adbfb43",
            "a1b1f4c47f487253e1ac88947b68b3b8")
    }

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        src_filename, tgt_filename, src_data_hash, tgt_data_hash = self.SPLITS[
            mode]
        src_fullname = os.path.join(default_root, src_filename)
        tgt_fullname = os.path.join(default_root, tgt_filename)

        if (not os.path.exists(src_fullname) or
            (src_data_hash and not md5file(src_fullname) == src_data_hash)) or (
                not os.path.exists(tgt_fullname) or
                (tgt_data_hash and not md5file(tgt_fullname) == tgt_data_hash)):
            get_path_from_url(self.URL, default_root, self.MD5)

        return src_fullname, tgt_fullname

    def _read(self, filename):
        src_filename, tgt_filename = filename
        with open(src_filename, 'r', encoding='utf-8') as src_f:
            with open(tgt_filename, 'r', encoding='utf-8') as tgt_f:
                for src_line, tgt_line in zip(src_f, tgt_f):
                    src_line = src_line.strip()
                    tgt_line = tgt_line.strip()
                    if not src_line or not tgt_line:
                        break

                    yield {"source": src_line, "target": tgt_line}
