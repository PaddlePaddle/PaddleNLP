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

import json
import os

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url

from ..utils.env import DATA_HOME
from .dataset import DatasetBuilder

__all__ = ["BelleGroup"]


class BelleGroup(DatasetBuilder):
    """
    From https://github.com/LianjiaTech/BELLE/tree/main

    """

    BUILDER_CONFIGS = {
        "generated_chat_0.4M": {
            "url": "https://bj.bcebos.com/dataset/BelleGroup/{}.zip",
            "md5": "",
            "splits": {
                "train": [os.path.join("{}", "train.json"), ""],
                "dev": [os.path.join("{}", "dev.json"), ""],
            },
        },
        "school_math_0.25M": {
            "url": "https://bj.bcebos.com/dataset/BelleGroup/{}.zip",
            "md5": "",
            "splits": {
                "train": [os.path.join("{}", "train.json"), ""],
                "dev": [os.path.join("{}", "dev.json"), ""],
            },
        },
        "train_2M_CN": {
            "url": "https://bj.bcebos.com/dataset/BelleGroup/{}.zip",
            "md5": "",
            "splits": {
                "train": [os.path.join("{}", "train.json"), ""],
                "dev": [os.path.join("{}", "dev.json"), ""],
            },
        },
        "train_1M_CN": {
            "url": "https://bj.bcebos.com/dataset/BelleGroup/{}.zip",
            "md5": "",
            "splits": {
                "train": [os.path.join("{}", "train.json"), ""],
                "dev": [os.path.join("{}", "dev.json"), ""],
            },
        },
        "train_0.5M_CN": {
            "url": "https://bj.bcebos.com/dataset/BelleGroup/{}.zip",
            "md5": "",
            "splits": {
                "train": [os.path.join("{}", "train.json"), ""],
                "dev": [os.path.join("{}", "dev.json"), ""],
            },
        },
        "multiturn_chat_0.8M": {
            "url": "https://bj.bcebos.com/dataset/BelleGroup/{}.zip",
            "md5": "",
            "splits": {
                "train": [os.path.join("{}", "train.json"), ""],
                "dev": [os.path.join("{}", "dev.json"), ""],
            },
        },
    }

    def _get_data(self, mode, **kwargs):
        builder_config = self.BUILDER_CONFIGS[self.name]

        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = builder_config["splits"][mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and not md5file(fullname) == data_hash):
            get_path_from_url(builder_config["url"], default_root, builder_config["md5"])

        return fullname

    def _read(self, filename, *args):
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                json_data = json.loads(line)

                yield {
                    "instruction": json_data["instruction"],
                    "input": json_data["input"],
                    "output": json_data["output"],
                }
