import collections
import json
import os

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME

from . import DatasetBuilder

__all__ = ['NLVR2']


class NLVR2(DatasetBuilder):
    URL_ANNO = "https://gitee.com/njcky/nlvr2/raw/master/annotations.tar.gz"
    MD5_ANNO = '319870abbc37d77cf394bcc3e9a4dd92'
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))

    SPLITS = {
        'train': META_INFO(
            os.path.join('annotations', 'train.json'),
            '8482649e6c7be8644ff6e6fe1c6f85d0'),
        'dev': META_INFO(
            os.path.join('annotations', 'dev.json'),
            'e0423ecad966f61a6548bacddb79edc8'),
        'test1': META_INFO(
            os.path.join('annotations', 'test1.json'),
            '9d9e0ff95ea7df582dde7721c8d77b3d'),
    }

    def _get_data(self, mode, **kwargs):
        # VQA annotations
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(self.URL_ANNO,
                              os.path.join(default_root, 'annotations'),
                              self.MD5_ANNO)

        return fullname

    def _read(self, filename, split):
        """Reads data."""

        with open(filename, 'r') as f:
            items = [json.loads(s) for s in f]

        default_root = os.path.join(DATA_HOME, self.__class__.__name__)

        for index, item in enumerate(items):
            sample = {}
            if item.get("label", None) is not None:
                sample["label"] = 1 if item["label"] == "True" else 0
            else:
                sample["label"] = 0  # Pseudo label

            sample["caption_a"] = item["sentence"]
            sample["identifier"] = item["identifier"]
            sample["feature_path_0"] = "{}img{}.png.npy".format(
                item["identifier"][:-1], 0)
            sample["feature_path_1"] = "{}img{}.png.npy".format(
                item["identifier"][:-1], 1)
            sample["split_name"] = split

            yield sample

    def get_labels(self):
        return ["True", "False"]

    def get_vocab(self):
        pass
