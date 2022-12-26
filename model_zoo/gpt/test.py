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

import os
import sys

import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(CURRENT_DIR)))


def setup():
    pass


def init_argv(config_name: str, config_file: str = "./configs/default.yaml"):
    """parse config file to argv

    Args:
        config_file (str, optional): the path of config file. Defaults to None.
    """
    # add tag if it's slow test
    if os.environ.get("slow_test", False):
        # eg: /path/to/file.json -> /path/to/file, .json
        config_file_name, file_suffix = os.path.splitext(config_file)

        # eg: /path/to/file.slow.json
        config_file_name, file_suffix = os.path.splitext(config_file)
        config_file = f"{config_file_name}.test{file_suffix}"

    config_file = os.path.join(CURRENT_DIR, config_file)

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)[config_name]

    argv = ["test.py"]
    for key, value in config.items():
        argv.append(f"--{key}")
        argv.append(str(value))
    sys.argv = argv


def test_pretrain():
    init_argv("pretrain", config_file="./configs/test.yaml")
    from run_pretrain_temp import do_train

    do_train()


def test_run_eval():
    init_argv("eval", config_file="./configs/test.yaml")
    from run_glue import do_train

    do_train()


def test_run_glue():
    init_argv("glue", config_file="./configs/test.yaml")
    from run_glue import do_train

    do_train()


def test_msra_ner():
    init_argv("msra_ner", config_file="./configs/test.yaml")
    from run_msra_ner import do_train

    do_train()


test_pretrain()
