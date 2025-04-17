# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import, division, print_function

import sys

from .device import get_device_and_mapping
from .log import logger


def check_version():
    """
    Log error and exit when the installed version of paddlepaddle is
    not satisfied.
    """
    err = (
        "PaddlePaddle version 1.8.0 or higher is required, "
        "or a suitable develop version is satisfied as well. \n"
        "Please make sure the version is good with your code."
    )
    try:
        pass
        # paddle.utils.require_version('0.0.0')
    except Exception:
        logger.error(err)
        sys.exit(1)


def check_device(device):
    """
    Log error and exit when using paddlepaddle cpu version.
    """
    err = (
        "You are using paddlepaddle %s version! Please try to \n"
        "1. install paddlepaddle-%s to run model on %s \nor 2. set the config option 'Global.device' to %s."
    )

    d, supported_device_map = get_device_and_mapping()

    assert (
        device in supported_device_map
    ), f"the device({device}) to check is not supported by now.Now the paddle only supports: {supported_device_map.keys()}"
    err = err % (d, device, device, d)

    try:
        assert supported_device_map[device]
    except AssertionError:
        logger.error(err)
        sys.exit(1)
