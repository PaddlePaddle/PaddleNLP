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
import os

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['CoteDp']


class CoteDp(DatasetBuilder):
    """
    COTE_DP dataset for Opinion Role Labeling task.
    More information please refer to https://aistudio.baidu.com/aistudio/competition/detail/50/?isFromLuge=1.

    """

    URL = "https://dataset-bj.cdn.bcebos.com/qianyan/COTE-DP.zip"
    MD5 = "a73d4170a283a2264a41c3ee9eb4d262"
    META_INFO = collections.namedtuple(
        'META_INFO', ('file', 'md5', 'field_indices', 'num_discarded_lines'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('COTE-DP', 'train.tsv'),
            '17d11ca91b7979f2c2023757650096e5', (0, 1), 1),
        'test': META_INFO(
            os.path.join('COTE-DP', 'test.tsv'),
            '5bb9b9ccaaee6bcc1ac7a6c852b46f66', (1, ), 1),
    }

    def _get_data(self, mode, **kwargs):
        """Downloads dataset."""
        default_root = os.path.join(DATA_HOME, 'COTE-DP')
        filename, data_hash, _, _ = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(self.URL, default_root, self.MD5)

        return fullname

    def _read(self, filename, split):
        """Reads data"""
        _, _, field_indices, num_discard_samples = self.SPLITS[split]
        with open(filename, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx < num_discard_samples:
                    continue
                line_stripped = line.strip().split('\t')
                if not line_stripped:
                    continue
                example = [line_stripped[indice] for indice in field_indices]
                if split == "test":
                    yield {"tokens": list(example[0])}
                else:
                    try:
                        entity, text = example[0], example[1]
                        start_idx = text.index(entity)
                    except:
                        # drop the dirty data
                        continue

                    labels = ['O'] * len(text)
                    labels[start_idx] = "B"
                    for idx in range(start_idx + 1, start_idx + len(entity)):
                        labels[idx] = "I"
                    yield {"tokens": list(text), "labels": labels}

    def get_labels(self):
        """
        Return labels of the COTE_BP.
        """
        return ["B", "I", "O"]
