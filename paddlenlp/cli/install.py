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

import os.path
import subprocess

from paddlenlp.utils.downloader import get_path_from_url_with_filelock, url_file_exists
from paddlenlp.utils.env import PACKAGE_HOME
from paddlenlp.utils.log import logger

PACKAGE_SERVER_HOME = "https://paddlenlp.bj.bcebos.com/wheels"


def install_package_from_bos(package_name: str, tag: str):
    """
    install package from bos server based on package_name and tag
    Args:
        package_name (str): the name of package, eg: paddlenlp, ppdiffusers, paddle-pipelines
        tag (str): pr number、 version of paddlenlp, or latest
    """
    # eg: https://paddlenlp.bj.bcebos.com/wheels/paddlenlp-latest-py3-none-any.whl
    file_name = f"{package_name}-{tag}-py3-none-any.whl"
    logger.info(f"start to downloading package<{file_name}>")

    package_url = f"{PACKAGE_SERVER_HOME}/{file_name}"
    if not url_file_exists(package_url):
        raise ValueError(f"there is not valid package<{package_name}_py3_{tag}.whl> " f"from the url<{package_url}>")

    file_path = os.path.join(PACKAGE_HOME, file_name)

    # force download
    file_path = get_path_from_url_with_filelock(package_url, PACKAGE_HOME, check_exist=False)

    # force reinstall the local package but ignore the dependencies
    command = f"python -m pip install --force-reinstall --no-dependencies {file_path}".split()
    subprocess.Popen(command)
