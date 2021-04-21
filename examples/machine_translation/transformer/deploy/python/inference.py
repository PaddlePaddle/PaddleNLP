import os
import sys

import argparse
import numpy as np
import yaml
from attrdict import AttrDict
from pprint import pprint

import paddle
from paddle import inference

from utils.recorder import Recorder

sys.path.append("../../")
import reader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, help="Batch size. ")
    parser.add_argument(
        "--config",
        default="./configs/transformer.big.yaml",
        type=str,
        help="Path of the config file. ")
    parser.add_argument(
        "--use-gpu", action="store_true", help="Whether to use gpu. ")
    parser.add_argument(
        "--use-xpu", action="store_true", help="Whether to use xpu. ")
    parser.add_argument(
        "--use-mkl", action="store_true", help="Whether to use mkl. ")
    parser.add_argument(
        "--threads",
        default=1,
        type=int,
        help="The number of threads when enable mkl. ")
    parser.add_argument(
        "--model-dir", default="", type=str, help="Path of the model. ")
    parser.add_argument(
        "--profile", action="store_true", help="Whether to profile. ")
    args = parser.parse_args()
    return args


def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    """
    Post-process the decoded sequence.
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


class Predictor(object):
    def __init__(self, predictor, input_handles, output_handles, recorder=None):
        self.predictor = predictor
        self.input_handles = input_handles
        self.output_handles = output_handles
        self.recorder = recorder

    @classmethod
    def create_predictor(cls, args, config=None, profile=False,
                         model_name=None):
        if config is None:
            config = inference.Config(
                os.path.join(args.inference_model_dir, "transformer.pdmodel"),
                os.path.join(args.inference_model_dir, "transformer.pdiparams"))
            if args.use_gpu:
                config.enable_use_gpu(100, 0)
            elif args.use_xpu:
                config.enable_xpu(100)
            else:
                # CPU
                config.disable_gpu()
                if args.use_mkl:
                    config.enable_mkldnn()
                    config.set_cpu_math_library_num_threads(args.threads)
            # Use ZeroCopy.
            config.switch_use_feed_fetch_ops(False)

        if profile:
            recorder = Recorder(config, args.infer_batch_size, model_name)
        else:
            recorder = None

        predictor = inference.create_predictor(config)
        input_handles = [
            predictor.get_input_handle(name)
            for name in predictor.get_input_names()
        ]
        output_handles = [
            predictor.get_input_handle(name)
            for name in predictor.get_output_names()
        ]
        return cls(predictor, input_handles, output_handles, recorder)

    def predict_batch(self, data):
        for input_field, input_handle in zip(data, self.input_handles):
            input_handle.copy_from_cpu(input_field.numpy() if isinstance(
                input_field, paddle.Tensor) else input_field)
        self.predictor.run()
        output = [
            output_handle.copy_to_cpu() for output_handle in self.output_handles
        ]
        return output

    def predict(self, test_loader, to_tokens, n_best, bos_idx, eos_idx):
        outputs = []
        samples = 0
        if self.recorder is not None:
            self.recorder.tic()

        for data in test_loader:
            samples += len(data[0])
            output = self.predict_batch(data)
            finished_sequence = output[0].transpose([0, 2, 1])
            for ins in finished_sequence:
                n_best_seq = []
                for beam_idx, beam in enumerate(ins):
                    if beam_idx >= n_best:
                        break
                    id_list = post_process_seq(beam, bos_idx, eos_idx)
                    word_list = to_tokens(id_list)
                    sequence = " ".join(word_list)
                    n_best_seq.append(sequence)
                outputs.append(n_best_seq)

        if self.recorder is not None:
            self.recorder.toc(samples)
            self.recorder.report()
        return outputs


def do_inference(args):
    # Define data loader
    test_loader, to_tokens = reader.create_infer_loader(args, True)

    predictor = Predictor.create_predictor(
        args=args, profile=args.profile, model_name=args.model_name)
    sequence_outputs = predictor.predict(test_loader, to_tokens, args.n_best,
                                         args.bos_idx, args.eos_idx)

    f = open(args.output_file, "w")
    for target in sequence_outputs:
        for sequence in target:
            f.write(sequence + "\n")
    f.close()


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)
    args.use_gpu = ARGS.use_gpu
    args.use_xpu = ARGS.use_xpu
    args.use_mkl = ARGS.use_mkl
    args.threads = ARGS.threads
    if ARGS.batch_size is not None:
        args.infer_batch_size = ARGS.batch_size
    args.profile = ARGS.profile
    args.model_name = "transformer_base" if "base" in ARGS.config else "transformer_big"
    if ARGS.model_dir != "":
        args.inference_model_dir = ARGS.model_dir

    do_inference(args)
