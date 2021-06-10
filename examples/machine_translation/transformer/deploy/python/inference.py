import os
import sys

import argparse
import numpy as np
import yaml
from attrdict import AttrDict
from pprint import pprint

import paddle
from paddle import inference

sys.path.append("../../")
import reader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="Batch size. ")
    parser.add_argument(
        "--config",
        default="./configs/transformer.big.yaml",
        type=str,
        help="Path of the config file. ")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["gpu", "xpu", "cpu"],
        help="Device to use during inference. ")
    parser.add_argument(
        "--use_mkl", action="store_true", help="Whether to use mkl. ")
    parser.add_argument(
        "--threads",
        default=1,
        type=int,
        help="The number of threads when enable mkl. ")
    parser.add_argument(
        "--model_dir", default="", type=str, help="Path of the model. ")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Whether to print logs on each cards and use benchmark vocab. Normally, not necessary to set --benchmark. "
    )
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
            if args.device == "gpu":
                config.enable_use_gpu(100, 0)
            elif args.device == "xpu":
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
            predictor.get_output_handle(name)
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
            cpu_rss_mb, gpu_rss_mb = 0, 0
            gpu_id = 0
            gpu_util = 0
            self.recorder.tic()

        for data in test_loader:
            samples += len(data[0])
            output = self.predict_batch(data)

            if self.recorder is not None:
                cm, gm = Recorder.get_current_memory_mb(gpu_id)
                cpu_rss_mb += cm
                gpu_rss_mb += gm
                gpu_util += Recorder.get_current_gputil(gpu_id)

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
            self.recorder.get_device_info(
                cpu_rss_mb=cpu_rss_mb / len(test_loader),
                gpu_rss_mb=gpu_rss_mb / len(test_loader),
                gpu_util=gpu_util / len(test_loader))
            self.recorder.report()
        return outputs


def do_inference(args):
    # Define data loader
    test_loader, to_tokens = reader.create_infer_loader(args)

    predictor = Predictor.create_predictor(
        args=args, profile=args.profile, model_name=args.model_name)
    sequence_outputs = predictor.predict(test_loader, to_tokens, args.n_best,
                                         args.bos_idx, args.eos_idx)

    f = open(args.output_file, "w", encoding="utf-8")
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
    args.benchmark = ARGS.benchmark
    args.device = ARGS.device
    args.use_mkl = ARGS.use_mkl
    args.threads = ARGS.threads
    if ARGS.batch_size is not None:
        args.infer_batch_size = ARGS.batch_size
    args.profile = ARGS.profile
    args.model_name = "transformer_base" if "base" in ARGS.config else "transformer_big"
    if ARGS.model_dir != "":
        args.inference_model_dir = ARGS.model_dir

    if args.profile:
        from utils.recorder import Recorder

    do_inference(args)
