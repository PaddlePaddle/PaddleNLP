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

from args import init_argv

DEFAULT_CONFIG_FILE = "./configs/test.yaml"


def test_pretrain():
    init_argv("pretrain", DEFAULT_CONFIG_FILE)
    from run_pretrain import do_train

    do_train()


def test_run_eval():
    init_argv("eval", DEFAULT_CONFIG_FILE)
    from run_glue import do_train

    do_train()


def test_run_glue():
    init_argv("glue", DEFAULT_CONFIG_FILE)
    from run_glue import do_train

    do_train()


def test_msra_ner():
    init_argv("msra_ner", DEFAULT_CONFIG_FILE)
    from run_msra_ner import do_train

    do_train()


def test_generation():
    # do not test under the slow_test
    if not os.getenv("slow_test", False):
        return

    init_argv("generation", DEFAULT_CONFIG_FILE)
    from run_generation import run

    run()


# you can uncomment the following code to debug your application in local IDE
# test_msra_ner()
