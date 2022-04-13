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
from pprint import pprint

import paddle
import paddle.inference as paddle_infer

from paddlenlp.transformers import Ernie3Tokenizer, Ernie3ForGeneration
from paddlenlp.ops.ext_utils import load
from paddlenlp.utils.log import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="ernie3-10b",
        type=str,
        help="The model name to specify the ernie3 to use. ")
    parser.add_argument(
        "--topk",
        default=4,
        type=int,
        help="The number of candidate to procedure topk sampling. ")
    parser.add_argument(
        "--topp",
        default=0.0,
        type=float,
        help="The probability threshold to procedure topp sampling. ")
    parser.add_argument(
        "--min_out_len", default=10, type=int, help="Minimum output length. ")
    parser.add_argument(
        "--max_out_len", default=20, type=int, help="Maximum output length. ")
    parser.add_argument(
        "--num_return_sequences",
        default=1,
        type=int,
        help="The number of returned sequences. ")
    parser.add_argument(
        "--temperature",
        default=0.9,
        type=float,
        help="The temperature to set. ")
    parser.add_argument(
        "--repetition_penalty",
        default=1.1,
        type=float,
        help="The repetition_penalty to set. ")
    parser.add_argument(
        "--decoding_strategy",
        default="sampling",
        choices=["sampling", "greedy_search"],
        type=str,
        help="The main strategy to decode. ")
    parser.add_argument(
        "--use_fp32_decoding",
        action="store_true",
        help="Whether to use fp32 decoding to predict. ")
    args = parser.parse_args()
    return args


def tokenize_input(tokenizer, texts):
    input_ids = []
    max_len = 0
    for text in texts:
        ids = tokenizer(text)['input_ids']
        max_len = max(max_len, len(ids))
        input_ids.append(ids)
    for i in range(len(input_ids)):
        if len(input_ids[i]) < max_len:
            input_ids[i] += [tokenizer.pad_token_id] * (
                max_len - len(input_ids[i]))
    input_ids = paddle.to_tensor(input_ids, dtype="int32")
    return input_ids


def infer(args):
    place = "gpu"
    place = paddle.set_device(place)
    paddle.set_default_dtype('float16')

    tokenizer = Ernie3Tokenizer.from_pretrained(args.model_name_or_path)
    logger.info('Loading the model parameters, please wait...')
    model = Ernie3ForGeneration.from_pretrained(
        args.model_name_or_path, load_state_as_np=True)
    model.eval()

    texts = ["中国的首都是哪里"]
    input_ids = tokenize_input(tokenizer, texts)

    output_ids, _ = model.generate(
        input_ids=input_ids,
        top_k=args.topk,
        top_p=args.topp,
        eos_token_id=tokenizer.end_token_id,
        decode_strategy=args.decoding_strategy,
        min_length=args.min_out_len,
        max_length=args.max_out_len,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        use_fp16_decoding=not args.use_fp32_decoding,
        use_faster=True)

    for idx, out in enumerate(output_ids.numpy()):
        seq = tokenizer.convert_ids_to_string(out)
        print(f'{idx}: {seq}')


if __name__ == "__main__":
    args = parse_args()
    pprint(args)

    infer(args)
