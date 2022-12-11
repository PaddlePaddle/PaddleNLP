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

from paddlenlp.transformers import UnifiedTransformerLMHeadModel, UnifiedTransformerTokenizer
from paddlenlp.ops import FasterUnifiedTransformer

from paddlenlp.utils.log import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="plato-xl",
        type=str,
        help="The model name to specify the PLATO/UnifiedTransformer to use. ",
    )
    parser.add_argument(
        "--inference_model_dir", default="./infer_model/", type=str, help="Path to save inference model of gpt. "
    )
    parser.add_argument("--topk", default=4, type=int, help="The number of candidate to procedure top_k sampling. ")
    parser.add_argument(
        "--topp", default=1.0, type=float, help="The probability threshold to procedure top_p sampling. "
    )
    parser.add_argument("--max_out_len", default=64, type=int, help="Maximum output length. ")
    parser.add_argument("--min_out_len", default=1, type=int, help="Minimum output length. ")
    parser.add_argument("--temperature", default=1.0, type=float, help="The temperature to set. ")
    parser.add_argument("--num_return_sequences", default=1, type=int, help="The number of returned sequences. ")
    parser.add_argument("--use_fp16_decoding", action="store_true", help="Whether to use fp16 decoding to predict. ")
    parser.add_argument(
        "--decoding_strategy",
        default="sampling",
        choices=["sampling", "beam_search"],
        type=str,
        help="The main strategy to decode. ",
    )
    parser.add_argument("--num_beams", default=4, type=int, help="The number of candidate to procedure beam search. ")
    parser.add_argument(
        "--diversity_rate", default=0.0, type=float, help="The diversity rate to procedure beam search. "
    )

    args = parser.parse_args()
    return args


def do_predict(args):
    place = "gpu"
    place = paddle.set_device(place)

    if args.use_fp16_decoding and os.getenv("PPFG_QKV_MEM_OPT", "0") == "1":
        paddle.set_default_dtype("float16")

    model_name = "plato-xl"
    model = UnifiedTransformerLMHeadModel.from_pretrained(model_name, load_state_as_np=True)
    tokenizer = UnifiedTransformerTokenizer.from_pretrained(model_name)

    plato = FasterUnifiedTransformer(model=model, use_fp16_decoding=args.use_fp16_decoding)
    # Set evaluate mode
    plato.eval()

    # Convert dygraph model to static graph model
    plato = paddle.jit.to_static(
        plato,
        input_spec=[
            # input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int32"),
            # token_type_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int32"),
            # attention_mask
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype="float32"),
            # seq_len
            paddle.static.InputSpec(shape=[None], dtype="int32"),
            # role_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int32"),
            # position_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int32"),
            args.max_out_len,
            args.min_out_len,
            args.topk,
            args.topp,
            args.decoding_strategy,
            tokenizer.cls_token_id,  # cls/bos
            tokenizer.sep_token_id,  # sep/eos
            tokenizer.pad_token_id,  # pad
            args.num_beams,  # num_beams. Used for beam_search.
            args.diversity_rate,  # diversity rate. Used for beam search.
            args.temperature,
            args.num_return_sequences,
        ],
    )

    # Save converted static graph model
    paddle.jit.save(plato, os.path.join(args.inference_model_dir, "plato"))
    logger.info("PLATO has been saved to {}.".format(args.inference_model_dir))


if __name__ == "__main__":
    args = parse_args()
    pprint(args)

    do_predict(args)
