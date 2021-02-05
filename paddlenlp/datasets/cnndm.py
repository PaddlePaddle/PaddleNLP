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
from paddle.dataset.common import DATA_HOME, md5file
from paddle.utils.download import get_path_from_url

from .dataset import TSVDataset

__all__ = ['CnnDm']


class CnnDm(TSVDataset):
    URL = "https://ernie-github.cdn.bcebos.com/data-cnndm.tar.gz"
    MD5 = None
    SEGMENT_INFO = collections.namedtuple(
        'SEGMENT_INFO', ('file', 'md5', 'field_indices', 'num_discard_samples'))
    SEGMENTS = {
        'train': SEGMENT_INFO(
            os.path.join('cnndm', 'train', '1'),
            '8b10ed0ae31e71e8cd9105a6978d8970', (1, 2), 0),
        'dev': SEGMENT_INFO(
            os.path.join('cnndm', 'dev', '1'),
            '7cb22f9cac04a285790a91cebba75260', (1, 2), 0),
    }

    def __init__(self, segment='train', root=None, **kwargs):
        default_root = os.path.join(DATA_HOME)
        filename, data_hash, field_indices, num_discard_samples = self.SEGMENTS[
            segment]
        fullname = os.path.join(default_root,
                                filename) if root is None else os.path.join(
                                    os.path.expanduser(root), filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            if root is not None:  # not specified, and no need to warn
                warnings.warn(
                    'md5 check failed for {}, download {} data to {}'.format(
                        filename, self.__class__.__name__, default_root))
            path = get_path_from_url(self.URL, default_root, self.MD5)
            fullname = os.path.join(default_root, filename)
        super(CnnDm, self).__init__(
            fullname,
            field_indices=field_indices,
            num_discard_samples=num_discard_samples,
            **kwargs)
