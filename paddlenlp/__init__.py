# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

__version__ = '2.2.0'  # Maybe dev is better
import sys
if 'datasets' in sys.modules.keys():
    from paddlenlp.utils.log import logger
    logger.warning(
        "datasets module loaded before paddlenlp. "
        "This may cause PaddleNLP datasets to be unavalible in intranet.")
from . import data
from . import datasets
from . import embeddings
from . import ops
from . import layers
from . import metrics
from . import seq2vec
from . import transformers
from . import utils
from . import losses
from . import experimental
from .taskflow import Taskflow
from . import trainer
import paddle

paddle.disable_signal_handler()
