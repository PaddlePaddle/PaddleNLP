# -*- coding: utf_8 -*-
import os
import sys
sys.path.append("../")

import paddle
import paddle.fluid as fluid
import numpy as np

from models.model_check import check_cuda
from config import PDConfig
from run_classifier import create_model
import utils
import reader

def do_save_inference_model(args):
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
            infer_pyreader, probs, feed_target_names = create_model(
                args,
                pyreader_name='infer_reader',
                num_labels=args.num_labels,
                is_prediction=True)

    test_prog = test_prog.clone(for_test=True)
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

def inference(exe, test_program, test_pyreader, fetch_list, infer_phrase):
    """
    Inference Function
    """
    print("=================")
    test_pyreader.start()
    while True:
        try:
            np_props = exe.run(program=test_program, fetch_list=fetch_list, return_numpy=True)
            for probs in np_props[0]:
                print("%d\t%f\t%f" % (np.argmax(probs), probs[0], probs[1]))
        except fluid.core.EOFException:
            test_pyreader.reset()
            break

def test_inference_model(args):
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
            infer_pyreader, probs, feed_target_names = create_model(
                args,
                pyreader_name='infer_reader',
                num_labels=args.num_labels,
                is_prediction=True)

    test_prog = test_prog.clone(for_test=True)
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    processor = reader.SentaProcessor(data_dir=args.data_dir,
        vocab_path=args.vocab_path,
        random_seed=args.random_seed,
        max_seq_len=args.max_seq_len)

    num_labels = len(processor.get_labels())

    assert (args.inference_model_dir)
    infer_program, feed_names, fetch_targets = fluid.io.load_inference_model(
        dirname=args.inference_model_dir,
        executor=exe,
        model_filename="model.pdmodel",
        params_filename="params.pdparams")

    infer_data_generator = processor.data_generator(
        batch_size=args.batch_size/dev_count,
        phase="infer",
        epoch=1,
        shuffle=False)
    
    infer_pyreader.set_sample_list_generator(infer_data_generator)
    inference(exe, test_prog, infer_pyreader,
        [probs.name], "infer")

if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    args = PDConfig('senta_config.json')
    args.build()
    args.print_arguments()
    check_cuda(args.use_cuda)
    if args.do_save_inference_model:
        do_save_inference_model(args)
    else:
        test_inference_model(args)
