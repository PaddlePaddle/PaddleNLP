import collections
import json
import os

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['DuReaderRobust']


class DuReaderRobust(DatasetBuilder):
    URL = 'https://dataset-bj.cdn.bcebos.com/qianyan/dureader_robust-data.tar.gz'
    MD5 = None
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('dureader_robust-data', 'train.json'),
            '800a3dcb742f9fdf9b11e0a83433d4be'),
        'dev': META_INFO(
            os.path.join('dureader_robust-data', 'dev.json'),
            'ae73cec081eaa28a735204c4898a2222'),
        'test': META_INFO(
            os.path.join('dureader_robust-data', 'test.json'),
            'e0e8aa5c7b6d11b6fc3935e29fc7746f')
    }

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(self.URL, default_root)
            fullname = os.path.join(default_root, filename)

        return fullname

    def _read(self, filename):
        with open(filename, "r", encoding="utf8") as f:
            input_data = json.load(f)["data"]
        for entry in input_data:
            title = entry.get("title", "").strip()
            for paragraph in entry["paragraphs"]:
                context = paragraph["context"].strip()
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question = qa["question"].strip()
                    answer_starts = [
                        answer["answer_start"] for answer in qa["answers"]
                    ]
                    answers = [
                        answer["text"].strip() for answer in qa["answers"]
                    ]

                    yield {
                        'id': qas_id,
                        'title': title,
                        'context': context,
                        'question': question,
                        'answers': answers,
                        'answer_starts': answer_starts
                    }
