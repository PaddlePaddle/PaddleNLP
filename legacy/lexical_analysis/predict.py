# -*- coding: UTF-8 -*-
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

import argparse
import os
import time
import sys

import paddle.fluid as fluid
import paddle

import utils
import reader
import creator
sys.path.append('../shared_modules/models/')
from model_check import check_cuda
from model_check import check_version

parser = argparse.ArgumentParser(__doc__)
# 1. model parameters
model_g = utils.ArgumentGroup(parser, "model", "model configuration")
model_g.add_arg("word_emb_dim", int, 128,
                "The dimension in which a word is embedded.")
model_g.add_arg("grnn_hidden_dim", int, 256,
                "The number of hidden nodes in the GRNN layer.")
model_g.add_arg("bigru_num", int, 2,
                "The number of bi_gru layers in the network.")
model_g.add_arg("use_cuda", bool, False, "If set, use GPU for training.")

# 2. data parameters
data_g = utils.ArgumentGroup(parser, "data", "data paths")
data_g.add_arg("word_dict_path", str, "./conf/word.dic",
               "The path of the word dictionary.")
data_g.add_arg("label_dict_path", str, "./conf/tag.dic",
               "The path of the label dictionary.")
data_g.add_arg("word_rep_dict_path", str, "./conf/q2b.dic",
               "The path of the word replacement Dictionary.")
data_g.add_arg("infer_data", str, "./data/infer.tsv",
               "The folder where the training data is located.")
data_g.add_arg("init_checkpoint", str, "./model_baseline", "Path to init model")
data_g.add_arg(
    "batch_size", int, 200,
    "The number of sequences contained in a mini-batch, "
    "or the maximum number of tokens (include paddings) contained in a mini-batch."
)


def do_infer(args):
    dataset = reader.Dataset(args)

    infer_program = fluid.Program()
    with fluid.program_guard(infer_program, fluid.default_startup_program()):
        with fluid.unique_name.guard():

            infer_ret = creator.create_model(
                args, dataset.vocab_size, dataset.num_labels, mode='infer')
    infer_program = infer_program.clone(for_test=True)

    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    else:
        place = fluid.CPUPlace()

    pyreader = creator.create_pyreader(
        args,
        file_name=args.infer_data,
        feed_list=infer_ret['feed_list'],
        place=place,
        model='lac',
        reader=dataset,
        mode='infer')

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load model
    utils.init_checkpoint(exe, args.init_checkpoint, infer_program)

    result = infer_process(
        exe=exe,
        program=infer_program,
        reader=pyreader,
        fetch_vars=[infer_ret['words'], infer_ret['crf_decode']],
        dataset=dataset)
    for sent, tags in result:
        result_list = ['(%s, %s)' % (ch, tag) for ch, tag in zip(sent, tags)]
        print(''.join(result_list))


def infer_process(exe, program, reader, fetch_vars, dataset):
    """
    the function to execute the infer process
    :param exe: the fluid Executor
    :param program: the infer_program
    :param reader: data reader
    :return: the list of prediction result
    """

    def input_check(data):
        if data[0]['words'].lod()[0][-1] == 0:
            return data[0]['words']
        return None

    results = []
    for data in reader():
        crf_decode = input_check(data)
        if crf_decode:
            results += utils.parse_result(crf_decode, crf_decode, dataset)
            continue

        words, crf_decode = exe.run(
            program,
            fetch_list=fetch_vars,
            feed=data,
            return_numpy=False,
            use_program_cache=True, )
        results += utils.parse_result(words, crf_decode, dataset)
    return results


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    args = parser.parse_args()
    check_cuda(args.use_cuda)
    check_version()
    do_infer(args)
