# -*- coding: utf-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import os
import random
import io
import math

import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor

import reader

import sys
line_tok = '\n'
space_tok = ' '
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
    line_tok = u'\n'
    space_tok = u' '

import os

from args import *
#from . import lm_model
import logging
import pickle

from model import VAE
from reader import BOS_ID, EOS_ID, get_vocab


def infer():
    args = parse_args()

    num_layers = args.num_layers
    src_vocab_size = args.vocab_size
    tar_vocab_size = args.vocab_size
    batch_size = args.batch_size
    init_scale = args.init_scale
    max_grad_norm = args.max_grad_norm
    hidden_size = args.hidden_size
    attr_init = args.attr_init
    latent_size = 32

    if args.enable_ce:
        fluid.default_main_program().random_seed = 102
        framework.default_startup_program().random_seed = 102

    model = VAE(hidden_size,
                latent_size,
                src_vocab_size,
                tar_vocab_size,
                batch_size,
                num_layers=num_layers,
                init_scale=init_scale,
                attr_init=attr_init)

    beam_size = args.beam_size
    trans_res = model.build_graph(mode='sampling', beam_size=beam_size)
    # clone from default main program and use it as the validation program
    main_program = fluid.default_main_program()

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    dir_name = args.reload_model
    print("dir name", dir_name)
    dir_name = os.path.join(dir_name, "checkpoint")
    fluid.load(main_program, dir_name, exe)
    vocab, tar_id2vocab = get_vocab(args.dataset_prefix)
    infer_output = np.ones((batch_size, 1), dtype='int64') * BOS_ID

    fetch_outs = exe.run(feed={'tar': infer_output},
                         fetch_list=[trans_res.name],
                         use_program_cache=False)

    with io.open(args.infer_output_file, 'w', encoding='utf-8') as out_file:

        for line in fetch_outs[0]:
            end_id = -1
            if EOS_ID in line:
                end_id = np.where(line == EOS_ID)[0][0]
            new_line = [tar_id2vocab[e[0]] for e in line[1:end_id]]
            out_file.write(space_tok.join(new_line))
            out_file.write(line_tok)


def check_version():
    """
    Log error and exit when the installed version of paddlepaddle is
    not satisfied.
    """
    err = "PaddlePaddle version 1.6 or higher is required, " \
          "or a suitable develop version is satisfied as well. \n" \
          "Please make sure the version is good with your code." \

    try:
        fluid.require_version('1.6.0')
    except Exception as e:
        logger.error(err)
        sys.exit(1)


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    check_version()
    infer()
