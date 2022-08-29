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

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['Tatoeba']

BASE_URL = "https://bj.bcebos.com/paddlenlp/datasets/tatoeba"

LANGUAGE_MAPPING = {
    "ar": "ara",
    "bg": "bul",
    "de": "deu",
    "el": "ell",
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "hi": "hin",
    "ru": "rus",
    "sw": "swh",
    "th": "tha",
    "tr": "tur",
    "ur": "urd",
    "vi": "vie",
    "zh": "cmn",
}


class Tatoeba(DatasetBuilder):
    '''
    The Tatoeba dataset, a collection of parallel sentences in multiple languages.
    '''
    META_INFO = collections.namedtuple(
        'META_INFO', ('file1', 'file2', 'md51', 'md52', 'URL1', 'URL2'))

    SPLITS = {
        "ara-eng":
        META_INFO(
            os.path.join(f"tatoeba.ara-eng.ara"),
            os.path.join(f"tatoeba.ara-eng.eng"),
            None,
            None,
            f"{BASE_URL}/tatoeba.ara-eng.ara",
            f"{BASE_URL}/tatoeba.ara-eng.eng",
        ),
        "bul-eng":
        META_INFO(
            os.path.join(f"tatoeba.bul-eng.bul"),
            os.path.join(f"tatoeba.bul-eng.eng"),
            None,
            None,
            f"{BASE_URL}/tatoeba.bul-eng.bul",
            f"{BASE_URL}/tatoeba.bul-eng.eng",
        ),
        "deu-eng":
        META_INFO(
            os.path.join(f"tatoeba.deu-eng.deu"),
            os.path.join(f"tatoeba.deu-eng.eng"),
            None,
            None,
            f"{BASE_URL}/tatoeba.deu-eng.deu",
            f"{BASE_URL}/tatoeba.deu-eng.eng",
        ),
        "ell-eng":
        META_INFO(
            os.path.join(f"tatoeba.ell-eng.ell"),
            os.path.join(f"tatoeba.ell-eng.eng"),
            None,
            None,
            f"{BASE_URL}/tatoeba.ell-eng.ell",
            f"{BASE_URL}/tatoeba.ell-eng.eng",
        ),
        "spa-eng":
        META_INFO(
            os.path.join(f"tatoeba.spa-eng.spa"),
            os.path.join(f"tatoeba.spa-eng.eng"),
            None,
            None,
            f"{BASE_URL}/tatoeba.spa-eng.spa",
            f"{BASE_URL}/tatoeba.spa-eng.eng",
        ),
        "fra-eng":
        META_INFO(
            os.path.join(f"tatoeba.fra-eng.fra"),
            os.path.join(f"tatoeba.fra-eng.eng"),
            None,
            None,
            f"{BASE_URL}/tatoeba.fra-eng.fra",
            f"{BASE_URL}/tatoeba.fra-eng.eng",
        ),
        "hin-eng":
        META_INFO(
            os.path.join(f"tatoeba.hin-eng.hin"),
            os.path.join(f"tatoeba.hin-eng.eng"),
            None,
            None,
            f"{BASE_URL}/tatoeba.hin-eng.hin",
            f"{BASE_URL}/tatoeba.hin-eng.eng",
        ),
        "rus-eng":
        META_INFO(
            os.path.join(f"tatoeba.rus-eng.rus"),
            os.path.join(f"tatoeba.rus-eng.eng"),
            None,
            None,
            f"{BASE_URL}/tatoeba.rus-eng.rus",
            f"{BASE_URL}/tatoeba.rus-eng.eng",
        ),
        "swh-eng":
        META_INFO(
            os.path.join(f"tatoeba.swh-eng.swh"),
            os.path.join(f"tatoeba.swh-eng.eng"),
            None,
            None,
            f"{BASE_URL}/tatoeba.swh-eng.swh",
            f"{BASE_URL}/tatoeba.swh-eng.eng",
        ),
        "tha-eng":
        META_INFO(
            os.path.join(f"tatoeba.tha-eng.tha"),
            os.path.join(f"tatoeba.tha-eng.eng"),
            None,
            None,
            f"{BASE_URL}/tatoeba.tha-eng.tha",
            f"{BASE_URL}/tatoeba.tha-eng.eng",
        ),
        "tur-eng":
        META_INFO(
            os.path.join(f"tatoeba.tur-eng.tur"),
            os.path.join(f"tatoeba.tur-eng.eng"),
            None,
            None,
            f"{BASE_URL}/tatoeba.tur-eng.tur",
            f"{BASE_URL}/tatoeba.tur-eng.eng",
        ),
        "urd-eng":
        META_INFO(
            os.path.join(f"tatoeba.urd-eng.urd"),
            os.path.join(f"tatoeba.urd-eng.eng"),
            None,
            None,
            f"{BASE_URL}/tatoeba.urd-eng.urd",
            f"{BASE_URL}/tatoeba.urd-eng.eng",
        ),
        "vie-eng":
        META_INFO(
            os.path.join(f"tatoeba.vie-eng.vie"),
            os.path.join(f"tatoeba.vie-eng.eng"),
            None,
            None,
            f"{BASE_URL}/tatoeba.vie-eng.vie",
            f"{BASE_URL}/tatoeba.vie-eng.eng",
        ),
        "cmn-eng":
        META_INFO(
            os.path.join(f"tatoeba.cmn-eng.cmn"),
            os.path.join(f"tatoeba.cmn-eng.eng"),
            None,
            None,
            f"{BASE_URL}/tatoeba.cmn-eng.cmn",
            f"{BASE_URL}/tatoeba.cmn-eng.eng",
        ),
    }

    def _get_data(self, mode, **kwargs):
        """
        returns the local path to the foreign lang sentence data, and the local path to the english sentence data.
        """
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        file_lang, file_eng, hash_lang, hash_eng, URL_lang, URL_eng = self.SPLITS[
            mode]
        full_filename_lang = os.path.join(default_root, file_lang)
        full_filename_eng = os.path.join(default_root, file_eng)

        if not os.path.exists(full_filename_lang) or (
                hash_lang and not md5file(full_filename_lang) == hash_lang):
            get_path_from_url(URL_lang, default_root)
        if not os.path.exists(full_filename_eng) or (
                hash_eng and not md5file(full_filename_eng) == hash_eng):
            get_path_from_url(URL_eng, default_root)
        return full_filename_lang, full_filename_eng

    def _read(self, filename_tuples, split):
        filename_lang, filename_eng = filename_tuples
        with open(filename_lang, "r", encoding="utf-8") as f:
            lines_lang = f.readlines()

        with open(filename_eng, "r", encoding="utf-8") as f:
            lines_eng = f.readlines()

        for sent_lang, sent_eng in zip(lines_lang, lines_eng):
            yield {"lang": sent_lang.strip(), "eng": sent_eng.strip()}
