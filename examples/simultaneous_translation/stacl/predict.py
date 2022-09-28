# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import argparse
from pprint import pprint
import yaml
from attrdict import AttrDict

import paddle
from paddlenlp.transformers import position_encoding_init
import reader
from model import SimultaneousTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="./config/transformer.yaml",
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


def do_predict(args):
    if args.device == 'gpu':
        place = "gpu:0"
    elif args.device == 'xpu':
        place = "xpu:0"
    elif args.device == 'cpu':
        place = "cpu"

    paddle.set_device(place)

    # Define data loader
    test_loader, to_tokens = reader.create_infer_loader(args)

    # Define model
    transformer = SimultaneousTransformer(
        args.src_vocab_size, args.trg_vocab_size, args.max_length + 1,
        args.n_layer, args.n_head, args.d_model, args.d_inner_hid, args.dropout,
        args.weight_sharing, args.bos_idx, args.eos_idx, args.waitk)

    # Load the trained model
    assert args.init_from_params, (
        "Please set init_from_params to load the infer model.")

    model_dict = paddle.load(
        os.path.join(args.init_from_params, "transformer.pdparams"))

    # To avoid a longer length than training, reset the size of position
    # encoding to max_length
    model_dict["src_pos_embedding.pos_encoder.weight"] = position_encoding_init(
        args.max_length + 1, args.d_model)
    model_dict["trg_pos_embedding.pos_encoder.weight"] = position_encoding_init(
        args.max_length + 1, args.d_model)

    transformer.load_dict(model_dict)

    # Set evaluate mode
    transformer.eval()

    f = open(args.output_file, "w", encoding='utf8')

    with paddle.no_grad():
        for input_data in test_loader:
            (src_word, ) = input_data

            finished_seq, finished_scores = transformer.greedy_search(
                src_word, max_len=args.max_out_len, waitk=args.waitk)
            finished_seq = finished_seq.numpy()
            finished_scores = finished_scores.numpy()
            for idx, ins in enumerate(finished_seq):
                for beam_idx, beam in enumerate(ins):
                    if beam_idx >= args.n_best:
                        break
                    id_list = post_process_seq(beam, args.bos_idx, args.eos_idx)
                    word_list = to_tokens(id_list)
                    sequence = ' '.join(word_list) + "\n"
                    f.write(sequence)
    f.close()


if __name__ == "__main__":
    args = parse_args()
    yaml_file = args.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)

    do_predict(args)
