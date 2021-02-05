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
    parser.add_argument(
        "--config",
        default="./configs/transformer.big.yaml",
        type=str,
        help="Path of the config file. ")
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
    def __init__(self, predictor, input_handles, output_handles):
        self.predictor = predictor
        self.input_handles = input_handles
        self.output_handles = output_handles

    @classmethod
    def create_predictor(cls, args, config=None):
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
                # such as enable_mkldnn, set_cpu_math_library_num_threads
                config.disable_gpu()
            # Use ZeroCopy.
            config.switch_use_feed_fetch_ops(False)

        predictor = inference.create_predictor(config)
        input_handles = [
            predictor.get_input_handle(name)
            for name in predictor.get_input_names()
        ]
        output_handles = [
            predictor.get_input_handle(name)
            for name in predictor.get_output_names()
        ]
        return cls(predictor, input_handles, output_handles)

    def predict_batch(self, data):
        for input_field, input_handle in zip(data, self.input_handles):
            input_handle.copy_from_cpu(input_field.numpy() if isinstance(
                input_field, paddle.Tensor) else input_field)
        self.predictor.run()
        output = [
            output_handle.copy_to_cpu() for output_handle in self.output_handles
        ]
        return output

    def predict(self, test_loader):
        outputs = []
        for data in test_loader:
            output = self.predict_batch(data)
            outputs.append(output)
        return outputs


def do_inference(args):
    # Define data loader
    test_loader, to_tokens = reader.create_infer_loader(args)

    predictor = Predictor.create_predictor(args)
    sequence_outputs = predictor.predict(test_loader)

    f = open(args.output_file, "w")
    for finished_sequence in sequence_outputs:
        finished_sequence = finished_sequence[0].transpose([0, 2, 1])
        for ins in finished_sequence:
            for beam_idx, beam in enumerate(ins):
                if beam_idx >= args.n_best:
                    break
                id_list = post_process_seq(beam, args.bos_idx, args.eos_idx)
                word_list = to_tokens(id_list)
                sequence = " ".join(word_list) + "\n"
                f.write(sequence)


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)

    do_inference(args)
