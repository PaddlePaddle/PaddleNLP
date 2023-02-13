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
from pprint import pprint

import paddle

from paddlenlp.ops import FasterPegasus
from paddlenlp.transformers import (
    PegasusChineseTokenizer,
    PegasusForConditionalGeneration,
)
from paddlenlp.utils.log import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese",
        type=str,
        help="The model name to specify the Pegasus to use. ",
    )
    parser.add_argument(
        "--export_output_dir", default="./inference_model", type=str, help="Path to save inference model of Pegasus. "
    )
    parser.add_argument("--topk", default=4, type=int, help="The number of candidate to procedure top_k sampling. ")
    parser.add_argument(
        "--topp", default=1.0, type=float, help="The probability threshold to procedure top_p sampling. "
    )
    parser.add_argument("--max_out_len", default=64, type=int, help="Maximum output length. ")
    parser.add_argument("--min_out_len", default=1, type=int, help="Minimum output length. ")
    parser.add_argument("--num_return_sequence", default=1, type=int, help="The number of returned sequence. ")
    parser.add_argument("--temperature", default=1.0, type=float, help="The temperature to set. ")
    parser.add_argument("--num_return_sequences", default=1, type=int, help="The number of returned sequences. ")
    parser.add_argument("--use_fp16_decoding", action="store_true", help="Whether to use fp16 decoding to predict. ")
    parser.add_argument(
        "--decoding_strategy",
        default="beam_search",
        choices=["beam_search"],
        type=str,
        help="The main strategy to decode. ",
    )
    parser.add_argument("--num_beams", default=4, type=int, help="The number of candidate to procedure beam search. ")
    parser.add_argument(
        "--diversity_rate", default=0.0, type=float, help="The diversity rate to procedure beam search. "
    )
    parser.add_argument(
        "--length_penalty",
        default=0.0,
        type=float,
        help="The exponential penalty to the sequence length in the beam_search strategy. ",
    )

    args = parser.parse_args()
    return args


def do_predict(args):
    place = "gpu"
    place = paddle.set_device(place)

    model_name_or_path = args.model_name_or_path
    model = PegasusForConditionalGeneration.from_pretrained(model_name_or_path)
    tokenizer = PegasusChineseTokenizer.from_pretrained(model_name_or_path)

    pegasus = FasterPegasus(model=model, use_fp16_decoding=args.use_fp16_decoding, trans_out=True)

    # Set evaluate mode
    pegasus.eval()

    # Convert dygraph model to static graph model
    pegasus = paddle.jit.to_static(
        pegasus,
        input_spec=[
            # input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int32"),
            # encoder_output
            None,
            # seq_len
            None,
            # min_length
            args.min_out_len,
            # max_length
            args.max_out_len,
            # num_beams. Used for beam_search.
            args.num_beams,
            # decoding_strategy
            args.decoding_strategy,
            # decoder_start_token_id
            model.decoder_start_token_id,
            # bos_token_id
            tokenizer.bos_token_id,
            # eos_token_id
            tokenizer.eos_token_id,
            # pad_token_id
            tokenizer.pad_token_id,
            # diversity rate. Used for beam search.
            args.diversity_rate,
            # length_penalty
            args.length_penalty,
            # topk
            args.topk,
            # topp
            args.topp,
            # temperature
            args.temperature,
            # num_return_sequences
            args.num_return_sequences,
        ],
    )

    # Save converted static graph model
    paddle.jit.save(pegasus, os.path.join(args.export_output_dir, "pegasus"))
    logger.info("PEGASUS has been saved to {}.".format(args.export_output_dir))


if __name__ == "__main__":
    args = parse_args()
    pprint(args)

    do_predict(args)
