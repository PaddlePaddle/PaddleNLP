import collections
import json
import os

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['BQCorpus']


class BQCorpus(DatasetBuilder):
    """
    BQCorpus: the largest dataset available for for the banking and finance sector

    by frozenfish123@Wuhan University

    """
    lazy = False
    URL = "https://dataset-bj.cdn.bcebos.com/qianyan/bq_corpus.zip"
    MD5 = "abe6c480b96cb705b4d24bd522848009"
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('BQCorpus', 'train.tsv'),
            'd37683e9ee778ee2f4326033b654adb9'),
        'dev': META_INFO(
            os.path.join('BQCorpus', 'dev.tsv'),
            '8a71f2a69453646921e9ee1aa457d1e4'),
        'test': META_INFO(
            os.path.join('BQCorpus', 'test.tsv'),
            'c797995baa248b144ceaa4018b191e52'),
    }

    def _get_data(self, mode, **kwargs):
        ''' Check and download Dataset '''
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(self.URL, default_root, self.MD5)

        return fullname

    def _read(self, filename):
        """Reads data."""
        with open(filename, 'r', encoding='utf-8') as f:
            head = None
            for line in f:
                data = line.strip().split("\t")
                if not head:
                    head = data
                else:
                    sentence1, sentence2, label = data
                    yield {
                        "sentence1": sentence1,
                        "sentence2": sentence2,
                        "label": label
                    }

    def get_labels(self):
        """
        Return labels of the BQCorpus object.
        """
        return ["0", "1"]
