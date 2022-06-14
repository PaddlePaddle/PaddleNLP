import argparse
import yaml
from attrdict import AttrDict
from pprint import pprint

import sys
import numpy as np
import os

from paddle_serving_client import Client
try:
    from paddle_serving_server_gpu.web_service import WebService
except:
    from paddle_serving_server.web_service import WebService

from transformer_reader import TransformerReader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="../configs/transformer.big.yaml",
                        type=str,
                        help="Path of the config file. ")
    parser.add_argument("--device",
                        default="gpu",
                        type=str,
                        choices=["gpu", "cpu"],
                        help="Device to use during inference. ")
    parser.add_argument("--model_dir",
                        default="",
                        type=str,
                        help="Path of the model. ")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=
        "Whether to print logs on each cards and use benchmark vocab. Normally, not necessary to set --benchmark. "
    )
    parser.add_argument("--profile",
                        action="store_true",
                        help="Whether to profile. ")
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


class TransformerService(WebService):

    def init_client(self, args):
        self.args = args
        self.transformer_reader = TransformerReader(args=args)

    def preprocess(self, feed=[], fetch=[]):
        src_sequence = feed[0]["src_word"]
        if isinstance(src_sequence, str):
            src_sequence = [src_sequence]
        src_word = self.transformer_reader.prepare_infer_input(src_sequence)
        feed_batch = {"src_word": src_word}
        fetch = ["save_infer_model/scale_0.tmp_1"]

        return feed_batch, fetch, True

    def postprocess(self, feed={}, fetch=[], fetch_map=None):
        if fetch_map is not None:
            finished_sequence = np.array(
                fetch_map["save_infer_model/scale_0.tmp_1"]).transpose(
                    [0, 2, 1])
            outputs = []
            for ins in finished_sequence:
                n_best_seq = []
                for beam_idx, beam in enumerate(ins):
                    if beam_idx >= self.args.n_best:
                        break
                    id_list = post_process_seq(beam, self.args.bos_idx,
                                               self.args.eos_idx)
                    word_list = self.transformer_reader.to_tokens(id_list)
                    sequence = " ".join(word_list)
                    n_best_seq.append(sequence)
                outputs.append(n_best_seq)
            res = {"finished_sequence": outputs}
            return res


def do_server(args):
    service = TransformerService(name="transformer")
    if args.profile:
        try:
            service.setup_profile(30)
        except:
            pass
    service.load_model_config(args.inference_model_dir)
    if args.device == "gpu":
        service.set_gpus("0")
        service.prepare_server(workdir="workdir",
                               port=9292,
                               device="gpu",
                               gpuid=0)
    else:
        service.prepare_server(workdir="workdir", port=9292, device="cpu")

    service.init_client(args=args)

    if args.profile:
        service.run_debugger_service()
    else:
        service.run_rpc_service()
    service.run_web_service()


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)
    args.benchmark = ARGS.benchmark
    args.profile = ARGS.profile
    args.device = ARGS.device
    if ARGS.model_dir != "":
        args.inference_model_dir = ARGS.model_dir

    do_server(args)
