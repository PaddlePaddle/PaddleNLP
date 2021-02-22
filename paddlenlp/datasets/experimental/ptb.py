import os
import math
import collections

from paddle.io import Dataset

from paddle.utils.download import get_path_from_url
from paddle.dataset.common import md5file
from paddlenlp.utils.env import DATA_HOME
import paddle.distributed as dist
from . import DatasetBuilder
import numpy as np

__all__ = ['PTB']


class PTB(DatasetBuilder):
    URL = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'
    MD5 = None
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('simple-examples', 'data', 'ptb.train.txt'), None),
        'valid': META_INFO(
            os.path.join('simple-examples', 'data', 'ptb.valid.txt'), None),
        'test': META_INFO(
            os.path.join('simple-examples', 'data', 'ptb.test.txt'), None)
    }

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):

            get_path_from_url(self.URL, default_root, self.MD5)
            fullname = os.path.join(default_root, filename)

        return fullname

    def _read(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line_stripped = line.strip()
                yield {"sentence": line_stripped}
