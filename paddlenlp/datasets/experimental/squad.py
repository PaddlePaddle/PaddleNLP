import copy
import collections
import json
import os
import warnings

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddle.io import Dataset
from paddlenlp.utils.env import DATA_HOME
from paddlenlp.transformers.tokenizer_utils import _is_whitespace, _is_control, convert_to_unicode
from . import DatasetBuilder

__all__ = ['SQuAD']


class SQuAD(DatasetBuilder):
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5', 'URL'))

    SPLITS = {
        'train_v1': META_INFO(
            os.path.join('train-v1.1.json'), '981b29407e0affa3b1b156f72073b945',
            'https://paddlenlp.bj.bcebos.com/datasets/squad/train-v1.1.json'),
        'dev_v1': META_INFO(
            os.path.join('dev-v1.1.json'), '3e85deb501d4e538b6bc56f786231552',
            'https://paddlenlp.bj.bcebos.com/datasets/squad/dev-v1.1.json'),
        'train_v2': META_INFO(
            os.path.join('train-v2.0.json'), '62108c273c268d70893182d5cf8df740',
            'https://paddlenlp.bj.bcebos.com/datasets/squad/train-v2.0.json'),
        'dev_v2': META_INFO(
            os.path.join('dev-v2.0.json'), '246adae8b7002f8679c027697b0b7cf8',
            'https://paddlenlp.bj.bcebos.com/datasets/squad/dev-v2.0.json')
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
                    is_impossible = False

                    if "is_impossible" in qa.keys():
                        is_impossible = qa["is_impossible"]

                    if not is_impossible:
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
                        'answer_starts': answer_starts,
                        'is_impossible': is_impossible
                    }
