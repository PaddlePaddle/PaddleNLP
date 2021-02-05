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

import copy
import collections
import io
import os
import warnings

from paddle.io import Dataset
from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME

from .dataset import TSVDataset

__all__ = ['ChnSentiCorp']


class ChnSentiCorp(TSVDataset):
    """
    ChnSentiCorp (by Tan Songbo at ICT of Chinese Academy of Sciences, and for
    opinion mining)

    """

    URL = "https://bj.bcebos.com/paddlehub-dataset/chnsenticorp.tar.gz"
    MD5 = "fbb3217aeac76a2840d2d5cd19688b07"
    META_INFO = collections.namedtuple(
        'META_INFO', ('file', 'md5', 'field_indices', 'num_discard_samples'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('chnsenticorp', 'train.tsv'),
            '689360c4a4a9ce8d8719ed500ae80907', (1, 0), 1),
        'dev': META_INFO(
            os.path.join('chnsenticorp', 'dev.tsv'),
            '05e4b02561c2a327833e05bbe8156cec', (1, 0), 1),
        'test': META_INFO(
            os.path.join('chnsenticorp', 'test.tsv'),
            '917dfc6fbce596bb01a91abaa6c86f9e', (1, 0), 1)
    }

    def __init__(self,
                 mode='train',
                 root=None,
                 return_all_fields=False,
                 **kwargs):
        if return_all_fields:
            splits = copy.deepcopy(self.__class__.SPLITS)
            mode_info = list(splits[mode])
            mode_info[2] = None
            splits[mode] = self.META_INFO(*mode_info)
            self.SPLITS = splits

        self._get_data(root, mode, **kwargs)

    def _get_data(self, root, mode, **kwargs):
        default_root = DATA_HOME
        filename, data_hash, field_indices, num_discard_samples = self.SPLITS[
            mode]
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
        super(ChnSentiCorp, self).__init__(
            fullname,
            field_indices=field_indices,
            num_discard_samples=num_discard_samples,
            **kwargs)

    def get_labels(self):
        """
        Return labels of the ChnSentiCorp object.
        """
        return ["0", "1"]
