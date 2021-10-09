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

import argparse
import time
from pprint import pprint

import paddle
from paddlenlp.ops import FasterBART
from paddlenlp.transformers import BartForConditionalGeneration, BartTokenizer
from paddlenlp.data import Pad
from paddlenlp.utils.log import logger


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


def prepare_input(tokenizer, sentences, pad_id):
    word_pad = Pad(pad_id, dtype="int64")
    tokenized = tokenizer(sentences)
    inputs = word_pad([i["input_ids"] for i in tokenized])
    input_ids = paddle.to_tensor(inputs)
    return input_ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="bart-base",
        type=str,
        help="The model name to specify the bart to use. Can be one of ['bart-base', 'bart-large',]. "
    )
    parser.add_argument(
        "--batch_size", default=1, type=int, help="Batch size. ")
    parser.add_argument(
        "--decoding_strategy",
        default='beam_search',
        type=str,
        help="The decoding strategy. Can be one of [beam_search, beam_search_v2, topk_sampling, topp_sampling]"
    )
    parser.add_argument(
        "--beam_size",
        default=1,
        type=int,
        help="The parameters for beam search. ")
    parser.add_argument(
        "--topk",
        default=1,
        type=int,
        help="The number of candidate to procedure beam search. ")
    parser.add_argument(
        "--topp",
        default=0.0,
        type=float,
        help="The probability threshold to procedure topp sampling. ")
    parser.add_argument(
        "--max_out_len", default=50, type=int, help="Maximum output length. ")
    parser.add_argument(
        "--diversity_rate",
        default=0.0,
        type=float,
        help="The diversity of beam search. ")
    parser.add_argument(
        "--n_best",
        default=1,
        type=int,
        help="The number of decoded sentences to output. ")
    parser.add_argument(
        "--rel_len",
        action="store_true",
        help=" Indicating whether max_out_len in configurations is the length relative to \
            that of source text. Only works in `v2` temporarily.")
    parser.add_argument(
        "--alpha",
        default=0.6,
        type=float,
        help="The power number in length penalty calculation. Only works in `v2` temporarily."
    )
    parser.add_argument(
        "--use_fp16_decoding",
        action="store_true",
        help="Whether to use fp16 decoding to predict. ")
    parser.add_argument(
        "--decoding_lib",
        default="../../build/lib/libdecoding_op.so",
        type=str,
        help="Path of libdecoding_op.so. ")
    args = parser.parse_args()
    return args


def do_predict(args):
    place = "gpu"
    place = paddle.set_device(place)

    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    logger.info('Loading the model parameters, please wait...')
    model = BartForConditionalGeneration.from_pretrained(
        args.model_name_or_path)
    # Set evaluate mode
    model.eval()
    sentences = [
        "I love that girl, but <mask> does not <mask> me.",
        "She is so <mask> that I can not help glance at <mask>.",
        "Nothing's gonna <mask> my love for you.",
        "Drop everything now. Meet me in the pouring <mask>. Kiss me on the sidewalk.",
    ]

    bos_id = model.bart.config['bos_token_id']
    eos_id = model.bart.config['eos_token_id']
    pad_id = model.bart.config['pad_token_id']
    input_ids = prepare_input(tokenizer, sentences, pad_id)

    # Define model
    faster_bart = FasterBART(
        model=model,
        decoding_strategy=args.decoding_strategy,
        beam_size=args.beam_size,
        topk=args.topk,
        topp=args.topp,
        max_out_len=args.max_out_len,
        diversity_rate=args.diversity_rate,
        decoding_lib=args.decoding_lib,
        use_fp16_decoding=args.use_fp16_decoding,
        rel_len=args.rel_len,
        alpha=args.alpha)

    # Set evaluate mode
    faster_bart.eval()

    with paddle.no_grad():
        for i in range(100):
            # For warmup.
            if 50 == i:
                paddle.fluid.core._cuda_synchronize(place)
                start = time.perf_counter()
            finished_seq = faster_bart(input_ids)
        paddle.fluid.core._cuda_synchronize(place)
        logger.info("Average test time for decoding is %f ms" % (
            (time.perf_counter() - start) / 50 * 1000))

        # Output
        if args.decoding_strategy.startswith('beam_search'):
            finished_seq = finished_seq.numpy().transpose([1, 2, 0])
            for ins in finished_seq:
                for beam_idx, beam in enumerate(ins):
                    if beam_idx >= args.n_best:
                        break
                    generated_ids = post_process_seq(beam, bos_id, eos_id)
                    print(tokenizer.convert_ids_to_string(generated_ids))
        elif args.decoding_strategy in ['topk_sampling', 'topp_sampling']:
            finished_seq = finished_seq.numpy().transpose([1, 0])
            for ins in finished_seq:
                generated_ids = post_process_seq(ins, bos_id, eos_id)
                print(tokenizer.convert_ids_to_string(generated_ids))


if __name__ == "__main__":
    args = parse_args()
    pprint(args)
    do_predict(args)
