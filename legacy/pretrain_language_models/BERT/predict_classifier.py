#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Load classifier's checkpoint to do prediction or save inference model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import sys
if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf8')

import os
import time
import argparse
import numpy as np
import multiprocessing
import paddle.fluid as fluid

import reader.cls as reader
from model.bert import BertConfig
from model.classifier import create_model

from utils.args import ArgumentGroup, print_arguments
from utils.init import init_pretraining_params

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "options to init, resume and save model.")
model_g.add_arg("bert_config_path",             str,  None,  "Path to the json file for bert model config.")
model_g.add_arg("init_checkpoint",              str,  None,  "Init checkpoint to resume training from.")
model_g.add_arg("save_inference_model_path",    str,  None,  "If set, save the inference model to this path.")
model_g.add_arg("use_fp16",                     bool, False, "Whether to resume parameters from fp16 checkpoint.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options.")
data_g.add_arg("data_dir",      str,  None,  "Directory to test data.")
data_g.add_arg("vocab_path",    str,  None,  "Vocabulary path.")
data_g.add_arg("max_seq_len",   int,  128,   "Number of words of the longest seqence.")
data_g.add_arg("batch_size",    int,  32,    "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("in_tokens",     bool, False,
              "If set, the batch size will be the maximum number of tokens in one batch. "
              "Otherwise, it will be the maximum number of examples in one batch.")
data_g.add_arg("do_lower_case", bool, True,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda",          bool,   True,  "If set, use GPU for training.")
run_type_g.add_arg("task_name",         str,    None,
                   "The name of task to perform fine-tuning, should be in {'xnli', 'mnli', 'cola', 'mrpc'}.")
run_type_g.add_arg("do_prediction",     bool,   True,  "Whether to do prediction on test set.")

args = parser.parse_args()
# yapf: enable.

def main(args):
    bert_config = BertConfig(args.bert_config_path)
    bert_config.print_config()

    task_name = args.task_name.lower()
    processors = {
        'xnli': reader.XnliProcessor,
        'cola': reader.ColaProcessor,
        'mrpc': reader.MrpcProcessor,
        'mnli': reader.MnliProcessor,
    }

    processor = processors[task_name](data_dir=args.data_dir,
                                      vocab_path=args.vocab_path,
                                      max_seq_len=args.max_seq_len,
                                      do_lower_case=args.do_lower_case,
                                      in_tokens=False)
    num_labels = len(processor.get_labels())

    predict_prog = fluid.Program()
    predict_startup = fluid.Program()
    with fluid.program_guard(predict_prog, predict_startup):
        with fluid.unique_name.guard():
            predict_data_loader, probs, feed_target_names = create_model(
                args,
                bert_config=bert_config,
                num_labels=num_labels,
                is_prediction=True)

    predict_prog = predict_prog.clone(for_test=True)

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    place = fluid.CUDAPlace(0) if args.use_cuda == True else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(predict_startup)

    if args.init_checkpoint:
        init_pretraining_params(exe, args.init_checkpoint, predict_prog, args.use_fp16)
    else:
        raise ValueError("args 'init_checkpoint' should be set for prediction!")

    # Due to the design that ParallelExecutor would drop small batches (mostly the last batch)
    # So using ParallelExecutor may left some data unpredicted
    # if prediction of each and every example is needed, please use Executor instead
    predict_exe = fluid.ParallelExecutor(
        use_cuda=args.use_cuda, main_program=predict_prog)

    predict_data_loader.set_batch_generator(
        processor.data_generator(
            batch_size=args.batch_size, phase='test', epoch=1, shuffle=False))

    predict_data_loader.start()
    all_results = []
    time_begin = time.time()
    while True:
        try:
            results = predict_exe.run(fetch_list=[probs.name])
            all_results.extend(results[0])
        except fluid.core.EOFException:
            predict_data_loader.reset()
            break
    time_end = time.time()

    np.set_printoptions(precision=4, suppress=True)
    print("-------------- prediction results --------------")
    print("example_id\t" + '  '.join(processor.get_labels()))
    for index, result in enumerate(all_results):
        print(str(index) + '\t{}'.format(result))

    if args.save_inference_model_path:
        _, ckpt_dir = os.path.split(args.init_checkpoint.rstrip('/'))
        dir_name = ckpt_dir + '_inference_model'
        model_path = os.path.join(args.save_inference_model_path, dir_name)
        print("save inference model to %s" % model_path)
        fluid.io.save_inference_model(
            model_path,
            feed_target_names, [probs],
            exe,
            main_program=predict_prog)


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    print_arguments(args)
    main(args)
