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
            "url": "https://paddlenlp.bj.bcebos.com/datasets/BelleGroup/generated_chat_0.4M.zip",
            "md5": "9bb71d4f2aa99acede2a0c3a8e761905",
            "splits": {
                "train": [os.path.join("generated_chat_0.4M", "train.json"), "47ea511025fbda9ffd6e5178677bb027"],
                "dev": [os.path.join("generated_chat_0.4M", "dev.json"), "d7bd4b71cdb006b9de90ebb634ca1179"],
            },
        },
        "school_math_0.25M": {
            "url": "https://paddlenlp.bj.bcebos.com/datasets/BelleGroup/school_math_0.25M.zip",
            "md5": "10076cbdc0a7436d55481f0234db8609",
            "splits": {
                "train": [os.path.join("school_math_0.25M", "train.json"), "e5a36fc9deb015254686c51e21528683"],
                "dev": [os.path.join("school_math_0.25M", "dev.json"), "99e967c38e39ed919327c011d9f6288f"],
            },
        },
        "train_2M_CN": {
            "url": "https://paddlenlp.bj.bcebos.com/datasets/BelleGroup/train_2M_CN.zip",
            "md5": "da88aca71eb9f454fab39db6a7e851e6",
            "splits": {
                "train": [os.path.join("train_2M_CN", "train.json"), "83e2917701a31ecf5152e4e9f234fcd0"],
                "dev": [os.path.join("train_2M_CN", "dev.json"), "74f67f04e30896aeccc10930a7dc1f40"],
            },
        },
        "train_1M_CN": {
            "url": "https://paddlenlp.bj.bcebos.com/datasets/BelleGroup/train_1M_CN.zip",
            "md5": "65380b542e8ddb4db8f8d2be0f28795c",
            "splits": {
                "train": [os.path.join("train_1M_CN.zip", "train.json"), "489886aba320c74a1fdfad43c652635b"],
                "dev": [os.path.join("train_1M_CN.zip", "dev.json"), "7bbf382aeab89f4398b2beca984e20e8"],
            },
        },
        "train_0.5M_CN": {
            "url": "https://paddlenlp.bj.bcebos.com/datasets/BelleGroup/train_0.5M_CN.zip",
            "md5": "45be55109ca9595efa36eaaed7c475d3",
            "splits": {
                "train": [os.path.join("train_0.5M_CN.zip", "train.json"), "61dc155956622c8389265de33b439757"],
                "dev": [os.path.join("train_0.5M_CN.zip", "dev.json"), "72617388fbc4897cb2952df3e5303c2b"],
            },
        },
        "multiturn_chat_0.8M": {
            "url": "https://paddlenlp.bj.bcebos.com/datasets/BelleGroup/multiturn_chat_0.8M.zip",
            "md5": "974bc42c5920e5722146a89dce2b10cc",
            "splits": {
                "train": [os.path.join("multiturn_chat_0.8M", "train.json"), "27e3a7ecff0f4a199f6e7119909988e9"],
                "dev": [os.path.join("multiturn_chat_0.8M", "dev.json"), "8fec175ea5e71cc78498d8ca3c1d5e66"],
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
