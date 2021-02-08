# -*- coding: UTF-8 -*-
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
sys.path.append("../shared_modules/")
import paddle
import paddle.fluid as fluid
import numpy as np

from models.model_check import check_cuda
from config import PDConfig
from run_classifier import create_model
import utils


def do_save_inference_model(args):
    if args.use_cuda:
        dev_count = fluid.core.get_cuda_device_count()
        place = fluid.CUDAPlace(0)
    else:
        dev_count = int(os.environ.get('CPU_NUM', 1))
        place = fluid.CPUPlace()

    test_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()

    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():
            infer_loader, probs, feed_target_names = create_model(
                args, num_labels=args.num_labels, is_prediction=True)

    test_prog = test_prog.clone(for_test=True)
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    assert (args.init_checkpoint)

    if args.init_checkpoint:
        utils.init_checkpoint(exe, args.init_checkpoint, test_prog)

    fluid.io.save_inference_model(
        args.inference_model_dir,
        feeded_var_names=feed_target_names,
        target_vars=[probs],
        executor=exe,
        main_program=test_prog,
        model_filename="model.pdmodel",
        params_filename="params.pdparams")

    print("save inference model at %s" % (args.inference_model_dir))


def test_inference_model(args, texts):
    if args.use_cuda:
        dev_count = fluid.core.get_cuda_device_count()
        place = fluid.CUDAPlace(0)
    else:
        dev_count = int(os.environ.get('CPU_NUM', 1))
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)

    assert (args.inference_model_dir)
    infer_program, feed_names, fetch_targets = fluid.io.load_inference_model(
        dirname=args.inference_model_dir,
        executor=exe,
        model_filename="model.pdmodel",
        params_filename="params.pdparams")
    data = []
    seq_lens = []
    for query in texts:
        wids = utils.query2ids(args.vocab_path, query)
        wids, seq_len = utils.pad_wid(wids)
        data.append(wids)
        seq_lens.append(seq_len)
    data = np.array(data)
    seq_lens = np.array(seq_lens)

    pred = exe.run(infer_program,
                   feed={feed_names[0]: data,
                         feed_names[1]: seq_lens},
                   fetch_list=fetch_targets,
                   return_numpy=True)
    for probs in pred[0]:
        print("%d\t%f\t%f\t%f" %
              (np.argmax(probs), probs[0], probs[1], probs[2]))


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    args = PDConfig(json_file="./config.json")
    args.build()
    args.print_arguments()
    check_cuda(args.use_cuda)
    if args.do_save_inference_model:
        do_save_inference_model(args)
    else:
        texts = [u"我 讨厌 你 ， 哼哼 哼 。 。", u"我 喜欢 你 ， 爱 你 哟"]
        test_inference_model(args, texts)
