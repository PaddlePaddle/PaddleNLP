# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import importlib

import paddle

from paddlenlp.utils.log import logger

cuda_version = float(paddle.version.cuda())
SUPPORTED_SM_VERSIONS = {70, 75, 80, 86, 89, 90} if cuda_version >= 12.4 else {70, 75, 80, 86, 89}


def get_sm_version():
    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return cc


sm_version = get_sm_version()
if sm_version not in SUPPORTED_SM_VERSIONS:
    raise RuntimeError("Unsupported SM version")
module_name = f"paddlenlp_ops.sm{sm_version}"

try:
    module = importlib.import_module(module_name)
    globals().update(vars(module))
except ImportError:
    logger.WARNING(f"No {module_name} ")
