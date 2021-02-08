# -*- coding: utf_8 -*-
import os
import sys
sys.path.append("../shared_modules/")
sys.path.append("../shared_modules/models/classification")
import paddle
import paddle.fluid as fluid
import numpy as np

from models.model_check import check_cuda
from config import PDConfig
from run_ernie_classifier import create_model
import utils
import reader
from run_ernie_classifier import ernie_pyreader
from models.representation.ernie import ErnieConfig
from models.representation.ernie import ernie_encoder, ernie_encoder_with_paddle_hub
from preprocess.ernie import task_reader


def do_save_inference_model(args):

    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    if args.use_cuda:
        dev_count = fluid.core.get_cuda_device_count()
        place = fluid.CUDAPlace(0)
    else:
        dev_count = int(os.environ.get('CPU_NUM', 1))
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)

    test_prog = fluid.Program()
    startup_prog = fluid.Program()

    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():
            infer_pyreader, ernie_inputs, labels = ernie_pyreader(
                args, pyreader_name="infer_reader")

            if args.use_paddle_hub:
                embeddings = ernie_encoder_with_paddle_hub(ernie_inputs,
                                                           args.max_seq_len)
            else:
                embeddings = ernie_encoder(
                    ernie_inputs, ernie_config=ernie_config)

            probs = create_model(
                args, embeddings, labels=labels, is_prediction=True)
    test_prog = test_prog.clone(for_test=True)
    exe.run(startup_prog)

    assert (args.init_checkpoint)

    if args.init_checkpoint:
        utils.init_checkpoint(exe, args.init_checkpoint, test_prog)

    fluid.io.save_inference_model(
        args.inference_model_dir,
        feeded_var_names=[
            ernie_inputs["src_ids"].name, ernie_inputs["sent_ids"].name,
            ernie_inputs["pos_ids"].name, ernie_inputs["input_mask"].name,
            ernie_inputs["seq_lens"].name
        ],
        target_vars=[probs],
        executor=exe,
        main_program=test_prog,
        model_filename="model.pdmodel",
        params_filename="params.pdparams")

    print("save inference model at %s" % (args.inference_model_dir))


def inference(exe, test_program, test_pyreader, fetch_list, infer_phrase):
    """
    Inference Function
    """
    print("=================")
    test_pyreader.start()
    while True:
        try:
            np_props = exe.run(program=test_program,
                               fetch_list=fetch_list,
                               return_numpy=True)
            for probs in np_props[0]:
                print("%d\t%f\t%f" % (np.argmax(probs), probs[0], probs[1]))
        except fluid.core.EOFException:
            test_pyreader.reset()
            break


def test_inference_model(args):
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    if args.use_cuda:
        dev_count = fluid.core.get_cuda_device_count()
        place = fluid.CUDAPlace(0)
    else:
        dev_count = int(os.environ.get('CPU_NUM', 1))
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)

    reader = task_reader.ClassifyReader(
        vocab_path=args.vocab_path,
        label_map_config=args.label_map_config,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case,
        random_seed=args.random_seed)

    test_prog = fluid.Program()
    startup_prog = fluid.Program()

    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():
            infer_pyreader, ernie_inputs, labels = ernie_pyreader(
                args, pyreader_name="infer_pyreader")

            embeddings = ernie_encoder(ernie_inputs, ernie_config=ernie_config)
            probs = create_model(
                args, embeddings, labels=labels, is_prediction=True)

    test_prog = test_prog.clone(for_test=True)
    exe.run(startup_prog)

    assert (args.inference_model_dir)
    infer_data_generator = reader.data_generator(
        input_file=args.test_set,
        batch_size=args.batch_size / dev_count,
        phase="infer",
        epoch=1,
        shuffle=False)

    infer_program, feed_names, fetch_targets = fluid.io.load_inference_model(
        dirname=args.inference_model_dir,
        executor=exe,
        model_filename="model.pdmodel",
        params_filename="params.pdparams")

    infer_pyreader.set_batch_generator(infer_data_generator)
    inference(exe, test_prog, infer_pyreader, [probs.name], "infer")


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    args = PDConfig()
    args.build()
    args.print_arguments()
    check_cuda(args.use_cuda)
    if args.do_save_inference_model:
        do_save_inference_model(args)
    else:
        test_inference_model(args)
