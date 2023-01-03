# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import os
import sys

import yaml

DEFAULT_CONFIG_FILE = "./configs/test.yaml"


def init_argv(config_name: str, config_file: str):
    """parse config file to argv
    Args:
        config_file (str, optional): the path of config file. Defaults to None.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)[config_name]

    # TODO(wj-Mcat): get the name of running application
    argv = ["run_classifier.py"]
    for key, value in config.items():
        argv.append(f"--{key}")
        argv.append(str(value))
    sys.argv = argv


def test_cross_lingual_transfer():
    # do not test under the slow_test
    if os.getenv("slow_test", None):
        return
    init_argv("cross-lingual-transfer", DEFAULT_CONFIG_FILE)
    from run_classifier import do_train

    do_train()


def test_translate_train_all():
    # do not test under the slow_test
    if os.getenv("slow_test", None):
        return

    init_argv("translate-train-all", DEFAULT_CONFIG_FILE)
    from run_classifier import do_train

    do_train()
