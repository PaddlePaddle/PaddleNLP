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
from paddlenlp.transformers import Ernie3PromptForGeneration, Ernie3PromptPretrainedModel, Ernie3PromptModel, Ernie3PromptTokenizer
from paddlenlp.utils.log import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="ernie3-prompt",
        type=str,
        help="The model name to specify the ernie3 to use. ")
    parser.add_argument(
        "--min_out_len", default=0, type=int, help="Minimum output length. ")
    parser.add_argument(
        "--max_out_len", default=5, type=int, help="Maximum output length. ")
    parser.add_argument(
        "--num_return_sequences",
        default=1,
        type=int,
        help="The number of returned sequences. ")
    parser.add_argument(
        "--decoding_strategy",
        default="beam_search",
        choices=["beam_search"],
        type=str,
        help="The main strategy to decode. ")
    parser.add_argument(
        "--num_beams", default=2, type=int, help="The beam size. ")
    parser.add_argument(
        "--use_fp16_decoding",
        action="store_true",
        help="Whether to use fp16 decoding to predict. ")
    args = parser.parse_args()
    return args


def infer(args):
    place = "gpu"
    place = paddle.set_device(place)

    tokenizer = Ernie3PromptTokenizer.from_pretrained(args.model_name_or_path)
    logger.info('Loading the model parameters, please wait...')
    model = Ernie3PromptForGeneration.from_pretrained(args.model_name_or_path)
    model.eval()

    texts = ["嗜睡抑郁多梦入睡困难很"]
    encoded_inputs = tokenizer.gen_encode(texts)

    output_ids, scores = model.generate(
        input_ids=encoded_inputs['input_ids'],
        position_ids=encoded_inputs['position_ids'],
        pos_ids_extra=encoded_inputs['pos_ids_extra'],
        bos_token_id=tokenizer.start_token_id,
        eos_token_id=tokenizer.gend_token_id,
        decode_strategy=args.decoding_strategy,
        num_beams=args.num_beams,
        min_length=args.min_out_len,
        max_length=args.max_out_len,
        use_fp16_decoding=args.use_fp16_decoding,
        decoding_lib="/home/PaddleNLP_ernie3_prompt/paddlenlp/ops/build/lib/libdecoding_op.so",
        use_faster=True)

    for idx, out in enumerate(output_ids.numpy()):
        seq = tokenizer.convert_ids_to_string(out)
        print(f'{idx}: {seq}: {scores[idx]}')


if __name__ == "__main__":
    args = parse_args()
    pprint(args)

    infer(args)
