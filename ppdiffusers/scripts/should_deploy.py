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

import subprocess
import sys


def get_version_of_package(name: str) -> str:
    """get version of remote package,

    apdapted from: https://stackoverflow.com/a/58649262/6894382

    Args:
        name (str): the name of package

    Returns:
        str: the version of package
    """
    latest_version = str(
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '{}==random'.format(name)],
            capture_output=True,
            text=True))
    latest_version = latest_version[latest_version.find('(from versions:') +
                                    15:]
    latest_version = latest_version[:latest_version.find(')')]
    latest_version = latest_version.replace(' ', '').split(',')[-1]
    return latest_version


def read_version():
    """read version of ppdiffusers"""
    pass
