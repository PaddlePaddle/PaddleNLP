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

import logging
import os
import six
import sys
import time

import numpy as np
import paddle
import paddle.fluid as fluid

from utils.input_field import InputField
from utils.configure import PDConfig
from utils.check import check_gpu, check_version
from utils.load import load

# include task-specific libs
import desc
import reader
from transformer import create_net, position_encoding_init


def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    """
    Post-process the beam-search decoded sequence. Truncate from the first
    <eos> and remove the <bos> and <eos> tokens currently.
    """
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1]
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq


def do_predict(args):
    if args.use_cuda:
        dev_count = fluid.core.get_cuda_device_count()
        place = fluid.CUDAPlace(0)
    else:
        dev_count = int(os.environ.get('CPU_NUM', 1))
        place = fluid.CPUPlace()
    # define the data generator
    processor = reader.DataProcessor(
        fpattern=args.predict_file,
        src_vocab_fpath=args.src_vocab_fpath,
        trg_vocab_fpath=args.trg_vocab_fpath,
        token_delimiter=args.token_delimiter,
        use_token_batch=False,
        batch_size=args.batch_size,
        device_count=dev_count,
        pool_size=args.pool_size,
        sort_type=reader.SortType.NONE,
        shuffle=False,
        shuffle_batch=False,
        start_mark=args.special_token[0],
        end_mark=args.special_token[1],
        unk_mark=args.special_token[2],
        max_length=args.max_length,
        n_head=args.n_head)
    batch_generator = processor.data_generator(phase="predict", place=place)
    args.src_vocab_size, args.trg_vocab_size, args.bos_idx, args.eos_idx, \
        args.unk_idx = processor.get_vocab_summary()
    trg_idx2word = reader.DataProcessor.load_dict(
        dict_path=args.trg_vocab_fpath, reverse=True)

    test_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()

    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():

            # define input and reader

            input_field_names = desc.encoder_data_input_fields + desc.fast_decoder_data_input_fields
            input_descs = desc.get_input_descs(args.args)
            input_slots = [{
                "name":
                name,
                "shape":
                input_descs[name][0],
                "dtype":
                input_descs[name][1],
                "lod_level":
                input_descs[name][2] if len(input_descs[name]) > 2 else 0
            } for name in input_field_names]

            input_field = InputField(input_slots)
            input_field.build(build_pyreader=True)

            # define the network

            predictions = create_net(
                is_training=False, model_input=input_field, args=args)
            out_ids, out_scores = predictions

    # This is used here to set dropout to the test mode.
    test_prog = test_prog.clone(for_test=True)

    # prepare predicting

    ## define the executor and program for training

    exe = fluid.Executor(place)

    exe.run(startup_prog)
    assert (
        args.init_from_params), "must set init_from_params to load parameters"
    load(test_prog, os.path.join(args.init_from_params, "transformer"), exe)
    print("finish initing model from params from %s" % (args.init_from_params))

    # to avoid a longer length than training, reset the size of position encoding to max_length
    for pos_enc_param_name in desc.pos_enc_param_names:
        pos_enc_param = fluid.global_scope().find_var(
            pos_enc_param_name).get_tensor()

        pos_enc_param.set(
            position_encoding_init(args.max_length + 1, args.d_model), place)

    exe_strategy = fluid.ExecutionStrategy()
    # to clear tensor array after each iteration
    exe_strategy.num_iteration_per_drop_scope = 1
    compiled_test_prog = fluid.CompiledProgram(test_prog).with_data_parallel(
        exec_strategy=exe_strategy, places=place)

    f = open(args.output_file, "wb")
    # start predicting
    ## decorate the pyreader with batch_generator
    input_field.loader.set_batch_generator(batch_generator)
    input_field.loader.start()
    while True:
        try:
            seq_ids, seq_scores = exe.run(
                compiled_test_prog,
                fetch_list=[out_ids.name, out_scores.name],
                return_numpy=False)

            # How to parse the results:
            #   Suppose the lod of seq_ids is:
            #     [[0, 3, 6], [0, 12, 24, 40, 54, 67, 82]]
            #   then from lod[0]:
            #     there are 2 source sentences, beam width is 3.
            #   from lod[1]:
            #     the first source sentence has 3 hyps; the lengths are 12, 12, 16
            #     the second source sentence has 3 hyps; the lengths are 14, 13, 15
            hyps = [[] for i in range(len(seq_ids.lod()[0]) - 1)]
            scores = [[] for i in range(len(seq_scores.lod()[0]) - 1)]
            for i in range(len(seq_ids.lod()[0]) -
                           1):  # for each source sentence
                start = seq_ids.lod()[0][i]
                end = seq_ids.lod()[0][i + 1]
                for j in range(end - start):  # for each candidate
                    sub_start = seq_ids.lod()[1][start + j]
                    sub_end = seq_ids.lod()[1][start + j + 1]
                    hyps[i].append(b" ".join([
                        trg_idx2word[idx]
                        for idx in post_process_seq(
                            np.array(seq_ids)[sub_start:sub_end], args.bos_idx,
                            args.eos_idx)
                    ]))
                    scores[i].append(np.array(seq_scores)[sub_end - 1])
                    f.write(hyps[i][-1] + b"\n")
                    if len(hyps[i]) >= args.n_best:
                        break
        except fluid.core.EOFException:
            break

    f.close()


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    args = PDConfig(yaml_file="./transformer.yaml")
    args.build()
    args.Print()
    check_gpu(args.use_cuda)
    check_version()

    do_predict(args)
