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
import io
import os
import string

import numpy as np

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['Imdb']


class Imdb(DatasetBuilder):
    """
    Implementation of `IMDB <https://www.imdb.com/interfaces/>`_ dataset.

    """
    URL = 'https://dataset.bj.bcebos.com/imdb%2FaclImdb_v1.tar.gz'
    MD5 = '7c2ac02c03563afcf9b574c7e56c153a'
    META_INFO = collections.namedtuple('META_INFO', ('file_dir', 'md5'))

    SPLITS = {
        'train': META_INFO(
            os.path.join('aclImdb', 'train'),
            '7c2ac02c03563afcf9b574c7e56c153a'),
        'test': META_INFO(
            os.path.join('aclImdb', 'test'),
            '7c2ac02c03563afcf9b574c7e56c153a'),
    }

    def _get_data(self, mode, **kwargs):
        """Downloads dataset."""
        default_root = DATA_HOME
        data_dir, data_hash = self.SPLITS[mode]
        data_dir = os.path.join(default_root, data_dir)
        tar_file = os.path.join(default_root, "imdb%2FaclImdb_v1.tar.gz")
        if not os.path.exists(tar_file) or (data_hash and
                                            not md5file(tar_file) == data_hash):
            path = get_path_from_url(self.URL, default_root, self.MD5)

        return data_dir

    def _read(self, data_dir):
        translator = str.maketrans('', '', string.punctuation)

        def _load_data(label):
            root = os.path.join(data_dir, label)
            data_files = os.listdir(root)

            if label == "pos":
                label_id = "1"
            elif label == "neg":
                label_id = "0"

            all_samples = []
            for f in data_files:
                f = os.path.join(root, f)
                data = io.open(f, 'r', encoding='utf8').readlines()
                data = data[0].translate(translator)
                all_samples.append((data, label_id))

            return all_samples

        data_set = _load_data("pos")
        data_set.extend(_load_data("neg"))
        np.random.shuffle(data_set)
        for data in data_set:
            yield {"text": data[0], "label": data[1]}

    def get_labels(self):
        """
        Return labels of the Imdb object.
        """
        return ["0", "1"]
