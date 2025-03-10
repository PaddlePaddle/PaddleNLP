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

import paddle
from ppfleetx.utils.log import logger


def version_check():
    version = paddle.version.full_version
    logger.info("run with paddle {}, commit id {}".format(paddle.__version__, paddle.__git_commit__[:8]))
    if version != "0.0.0":
        paddle.utils.require_version(min_version="2.4.0")
