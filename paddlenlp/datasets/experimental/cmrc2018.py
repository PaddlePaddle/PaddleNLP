import collections
import json
import os

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['CMRC2018']


class CMRC2018(DatasetBuilder):
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5', 'URL'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('cmrc2018_train.json'), None,
            'https://paddlenlp.bj.bcebos.com/datasets/cmrc/cmrc2018_train.json'),
        'dev': META_INFO(
            os.path.join('cmrc2018_dev.json'), None,
            'https://paddlenlp.bj.bcebos.com/datasets/cmrc/cmrc2018_dev.json'),
        'trial': META_INFO(
            os.path.join('cmrc2018_trial.json'), None,
            'https://paddlenlp.bj.bcebos.com/datasets/cmrc/cmrc2018_trial.json')
    }

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash, URL = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(URL, default_root)
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
