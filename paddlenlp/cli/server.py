# coding:utf-8
# copyright (c) 2022  paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license"
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

import uvicorn

from ..utils.log import logger


def start_backend(app, **kwargs):
    logger.info("The PaddleNLP SimpleServer is starting, backend component uvicorn arguments as follows:")
    for key, value in kwargs.items():
        if key != "log_config":
            logger.info("   the starting argument [{}]={}".format(key, value))
    uvicorn.run(app, **kwargs)
