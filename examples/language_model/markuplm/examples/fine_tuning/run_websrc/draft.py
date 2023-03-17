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

import argparse
import glob
import logging
import os
import random
import timeit

import numpy as np
import torch
from markuplmft.models.markuplm import (
    MarkupLMConfig,
    MarkupLMForQuestionAnswering,
    MarkupLMTokenizer,
    MarkupLMTokenizerFast,
)
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from utils import (
    RawResult,
    StrucDataset,
    convert_examples_to_features,
    read_squad_examples,
    write_predictions,
)
from utils_evaluate import EvalOpts
from utils_evaluate import main as evaluate_on_squad

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    mp = "../../../../../results/markuplm-base"
    op = "./moli"
    config = MarkupLMConfig.from_pretrained(mp)
    logger.info("=====Config for model=====")
    logger.info(str(config))
    max_depth = config.max_depth
    tokenizer = MarkupLMTokenizer.from_pretrained(mp)
    model = MarkupLMForQuestionAnswering.from_pretrained(mp, config=config)

    tokenizer.save_pretrained(op)
