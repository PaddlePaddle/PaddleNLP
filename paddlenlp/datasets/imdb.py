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

import io
import os
import string

import numpy as np
import paddle
from paddle.io import Dataset

from paddlenlp.utils.env import DATA_HOME
from paddlenlp.utils.downloader import get_path_from_url

__all__ = ['Imdb']


class Imdb(Dataset):
    """
    Implementation of `IMDB <https://www.imdb.com/interfaces/>`_ dataset.

    """
    URL = 'https://dataset.bj.bcebos.com/imdb%2FaclImdb_v1.tar.gz'
    MD5 = '7c2ac02c03563afcf9b574c7e56c153a'

    def __init__(
            self,
            root=None,
            mode='train', ):
        assert mode in [
            "train", "test"
        ], "Unknown mode %s, it should be 'train' or 'test'." % mode
        if root is None:
            root = DATA_HOME
        data_dir = os.path.join(root, "aclImdb")

        if not os.path.exists(data_dir):
            data_dir = get_path_from_url(self.URL, root, self.MD5)

        self.examples = self._read_data_file(data_dir, mode)

    def _read_data_file(self, data_dir, mode):
        # remove punctuations ad-hoc.
        translator = str.maketrans('', '', string.punctuation)

        def _load_data(label):
            root = os.path.join(data_dir, mode, label)
            data_files = os.listdir(root)
            data_files.sort()
            if label == "pos":
                label_id = 1
            elif label == "neg":
                label_id = 0

            all_samples = []
            for f in data_files:
                f = os.path.join(root, f)
                with io.open(f, 'r', encoding='utf8') as fr:
                    data = fr.readlines()
                    data = data[0].translate(translator)
                    all_samples.append((data, label_id))

            return all_samples

        data_set = _load_data("pos")
        data_set.extend(_load_data("neg"))
        np.random.shuffle(data_set)

        return data_set

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)

    def get_labels(self):
        return ["0", "1"]
