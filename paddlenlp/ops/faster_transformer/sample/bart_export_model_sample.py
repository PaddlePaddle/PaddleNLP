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

import os
import argparse
import paddle
from pprint import pprint
from paddlenlp.transformers import BartForConditionalGeneration, BartTokenizer
from paddlenlp.ops import FasterBART
from paddlenlp.utils.log import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",
                        default="bart-base",
                        type=str,
                        help="The model name to specify the bart to use. ")
    parser.add_argument("--inference_model_dir",
                        default="./infer_model/",
                        type=str,
                        help="Path to save inference model of bart. ")
    parser.add_argument(
        "--topk",
        default=4,
        type=int,
        help="The number of candidate to procedure top_k sampling. ")
    parser.add_argument(
        "--topp",
        default=1.0,
        type=float,
        help="The probability threshold to procedure top_p sampling. ")
    parser.add_argument("--max_out_len",
                        default=20,
                        type=int,
                        help="Maximum output length. ")
    parser.add_argument("--temperature",
                        default=1.0,
                        type=float,
                        help="The temperature to set. ")
    parser.add_argument("--num_return_sequences",
                        default=1,
                        type=int,
                        help="The number of returned sequences. ")
    parser.add_argument("--use_fp16_decoding",
                        action="store_true",
                        help="Whether to use fp16 decoding to predict. ")
    parser.add_argument("--decoding_strategy",
                        default="beam_search",
                        choices=["sampling", "beam_search"],
                        type=str,
                        help="The main strategy to decode. ")
    parser.add_argument(
        "--num_beams",
        default=5,
        type=int,
        help="The number of candidate to procedure beam search. ")
    parser.add_argument("--diversity_rate",
                        default=0.0,
                        type=float,
                        help="The diversity rate to procedure beam search. ")
    parser.add_argument("--repetition_penalty",
                        default=1.0,
                        type=float,
                        help="The repetition_penalty to set. ")
    parser.add_argument("--length_penalty",
                        default=0.0,
                        type=float,
                        help="The length penalty to decode. ")
    parser.add_argument("--early_stopping",
                        action="store_true",
                        help="Whether to do early stopping. ")

    args = parser.parse_args()
    return args


def do_predict(args):
    place = "gpu"
    place = paddle.set_device(place)

    model = BartForConditionalGeneration.from_pretrained(
        args.model_name_or_path)
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)

    # For opening faster_encoder
    model.eval()

    faster_bart = FasterBART(model=model,
                             use_fp16_decoding=args.use_fp16_decoding)
    # Set evaluate mode
    faster_bart.eval()

    # Convert dygraph model to static graph model
    faster_bart = paddle.jit.to_static(
        faster_bart,
        input_spec=[
            # input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int32"),
            # encoder_output
            None,
            # seq_len
            None,
            args.num_beams,  # num_beams.
            args.topk,
            args.topp,
            args.decoding_strategy,
            tokenizer.bos_token_id,  # bos
            tokenizer.eos_token_id,  # eos 
            tokenizer.pad_token_id,  # pad 
            tokenizer.eos_token_id,  # decoder_start_token_id
            args.max_out_len,  # max_length
            args.diversity_rate,  # diversity_rate
            args.length_penalty,  # length_penalty
            args.num_return_sequences,
            args.early_stopping,
            tokenizer.eos_token_id,  #forced_eos_token_id
        ])

    # Save converted static graph model
    paddle.jit.save(faster_bart, os.path.join(args.inference_model_dir, "bart"))
    logger.info("BART has been saved to {}.".format(args.inference_model_dir))


if __name__ == "__main__":
    args = parse_args()
    pprint(args)

    do_predict(args)
