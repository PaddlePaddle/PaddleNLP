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

import os, sys
import json

CURRENT_DIR = os.path.dirname(__file__)
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def init_argv(config_file: str = None):
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
        config_file = f'{config_file_name}.slow{file_suffix}'

    config_file = os.path.join(CURRENT_DIR, config_file)

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    argv = ['']
    for key, value in config.items():
        argv.append(f'--{key}')
        argv.append(str(value))
    sys.argv = argv


def test_pretrain():
    init_argv("./configs/pretrain.json")
    from run_pretrain import do_train
    do_train()


def test_run_glue():
    init_argv("./configs/glue.json")
    from run_glue import do_train
    do_train()


def test_msra_ner():
    init_argv("./configs/msra_ner.json")
    from run_msra_ner import do_train
    do_train()


test_msra_ner()
