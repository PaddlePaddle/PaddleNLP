# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
from pprint import pprint

import paddle
import paddle.inference as paddle_infer

from paddlenlp.transformers import BartForConditionalGeneration, BartTokenizer
from paddlenlp.ops.ext_utils import load


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_model_dir",
                        default="./infer_model/",
                        type=str,
                        help="Path to save inference model of BART. ")

    args = parser.parse_args()

    return args


def prepare_input(tokenizer, sentences):
    tokenized = tokenizer(sentences, padding=True)
    input_ids = np.asarray(tokenized['input_ids'], dtype="int32")
    return input_ids


def postprocess_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
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


def infer(args):
    model_name = 'bart-base'
    tokenizer = BartTokenizer.from_pretrained(model_name)

    sentences = [
        "I love that girl, but <mask> does not <mask> me.",
        "She is so <mask> that I can not help glance at <mask>.",
        "Nothing's gonna <mask> my love for you.",
        "Drop everything now. Meet me in the pouring <mask>. Kiss me on the sidewalk.",
    ]

    input_ids = prepare_input(tokenizer, sentences)

    # Load FasterTransformer lib.
    load("FasterTransformer", verbose=True)

    config = paddle_infer.Config(
        os.path.join(args.inference_model_dir, "bart.pdmodel"),
        os.path.join(args.inference_model_dir, "bart.pdiparams"))

    config.enable_use_gpu(100, 0)
    config.disable_glog_info()
    # `embedding_eltwise_layernorm_fuse_pass` failed
    config.delete_pass("embedding_eltwise_layernorm_fuse_pass")
    predictor = paddle_infer.create_predictor(config)

    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    input_handle.copy_from_cpu(input_ids.astype("int32"))

    predictor.run()

    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu()

    for idx, sample in enumerate(output_data.transpose([1, 2, 0]).tolist()):
        for beam_idx, beam in enumerate(sample):
            if beam_idx >= len(sample) / 2:
                break
            generated_ids = postprocess_seq(beam, tokenizer.bos_token_id,
                                            tokenizer.eos_token_id)
            seq = tokenizer.convert_ids_to_string(generated_ids)
            print(f'{idx}-{beam_idx}: {seq}')


if __name__ == "__main__":
    args = setup_args()
    pprint(args)

    infer(args)
