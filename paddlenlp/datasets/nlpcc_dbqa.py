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

from ..utils.env import DATA_HOME
from .dataset import DatasetBuilder

__all__ = ["NLPCC_DBQA"]


class NLPCC_DBQA(DatasetBuilder):
    """
    NLPCC2016 DBQA dataset.

    Document-based QA (or DBQA) task
    When predicting answers to each question, a DBQA system built by each
    participating team IS LIMITED TO select sentences as answers from the
    question’s given document.

    For more information: http://tcci.ccf.org.cn/conference/2016/dldoc/evagline2.pdf
    """

    URL = "https://bj.bcebos.com/paddlenlp/datasets/nlpcc-dbqa.zip"
    MD5 = "a5f69c2462136ef4d1707e4e2551a57b"
    META_INFO = collections.namedtuple("META_INFO", ("file", "md5"))
    SPLITS = {
        "train": META_INFO(os.path.join("nlpcc-dbqa", "nlpcc-dbqa", "train.tsv"), "4f84fefce1a8f52c8d9248d1ff5ab9bd"),
        "dev": META_INFO(os.path.join("nlpcc-dbqa", "nlpcc-dbqa", "dev.tsv"), "3831beb0d42c29615d06343538538f53"),
        "test": META_INFO(os.path.join("nlpcc-dbqa", "nlpcc-dbqa", "test.tsv"), "e224351353b1f6a15837008b5d0da703"),
    }

    def _get_data(self, mode, **kwargs):
        """Downloads dataset."""
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and not md5file(fullname) == data_hash):
            get_path_from_url(self.URL, default_root, self.MD5)

        return fullname

    def _read(self, filename, split):
        """Reads data."""
        with open(filename, "r", encoding="utf-8") as f:
            head = None
            for line in f:
                data = line.strip().split("\t")
                if not head:
                    head = data
                else:
                    qid, text_a, text_b, label = data
                    yield {"qid": qid, "text_a": text_a, "text_b": text_b, "label": label}

    def get_labels(self):
        """
        Return labels of XNLI dataset.

        Note:
            Contradictory and contradiction are the same label
        """
        return ["0", "1"]
