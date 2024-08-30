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
from datetime import datetime

PADDLENLP_STABLE_VERSION = "PADDLENLP_STABLE_VERSION"

# this version is used for develop and test.
# release version will be added fixed version by setup.py.
__version__ = "3.0.0b0.post"
if os.getenv(PADDLENLP_STABLE_VERSION):
    __version__ = __version__.replace(".post", "")
else:
    formatted_date = datetime.now().date().strftime("%Y%m%d")
    __version__ = __version__.replace(".post", ".post{}".format(formatted_date))

# follows version replaced by setup.py for release version.
# [VERSION_INFO]

if "datasets" in sys.modules.keys():
    from paddlenlp.utils.log import logger

    logger.warning(
        "Detected that datasets module was imported before paddlenlp. "
        "This may cause PaddleNLP datasets to be unavalible in intranet. "
        "Please import paddlenlp before datasets module to avoid download issues"
    )
import paddle

from . import (
    data,
    dataaug,
    datasets,
    embeddings,
    experimental,
    layers,
    losses,
    metrics,
    ops,
    peft,
    prompt,
    quantization,
    seq2vec,
    trainer,
    transformers,
    trl,
    utils,
    version,
)
from .server import SimpleServer
from .taskflow import Taskflow

paddle.disable_signal_handler()
