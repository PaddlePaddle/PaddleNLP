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

import time
import argparse
from pprint import pprint

from paddlenlp.transformers import UnifiedTransformerLMHeadModel, UnifiedTransformerTokenizer


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_role",
        action="store_true",
        help="Whether to use role embeddings. ")
    parser.add_argument(
        "--position_style",
        default="continuous",
        choice=["continuous", "relative"],
        type=str,
        help="The type for positional embedding. Default is continuous. ")
    parser.add_argument(
        "--max_out_len",
        default=64,
        type=int,
        help="Maximum output sequence length. ")
    parser.add_argument(
        "--min_out_len",
        default=1,
        type=int,
        help="Minimum output sequence length. ")
    parser.add_argument(
        "--topk",
        default=4,
        type=int,
        help="The k value for topk_sampling. Default is 4. ")
    parser.add_argument(
        "--topp",
        default=0.0,
        type=float,
        help="The p value for topp_sampling. Default is 0.0f. ")

    args = parse_args(parser)

    return args


def postprocess_response(token_ids, tokenizer):
    """Post-process the decoded sequence. Truncate from the first <eos>."""
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == tokenizer.sep_token_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = tokenizer.merge_subword(tokens)
    return tokens


def infer(args):
    model_name = 'plato-xl'
    model = UnifiedTransformerLMHeadModel.from_pretrained(model_name)
    tokenizer = UnifiedTransformerTokenizer.from_pretrained(model_name)

    context = [
        "Hi , Becky , what's up ?",
        "Not much , except that my mother-in-law is driving me up the wall .",
        "What's the problem ?"
    ]

    data = tokenizer.dialogue_encode(
        history=context,
        add_start_token_as_response=True,
        return_length=True,
        return_role_ids=args.use_role,
        position_style=args.position_style)

    for i in range(200):
        if 100 == i:
            paddle.device.cuda.synchronize()
            start = time.time()

        outputs, _ = model.generate(
            input_ids=data['input_ids'],
            token_type_ids=data['token_type_ids'],
            position_ids=data['position_ids'],
            attention_mask=data['attention_mask'],
            role_ids=data['role_ids'],
            seq_len=data['seq_len'],
            max_length=args.max_out_len,
            min_length=args.min_out_len,
            decode_strategy='sampling',
            top_k=args.topk,
            top_p=args.topp,
            use_fp16_decoding=True,
            use_faster=True)

    paddle.device.cuda.synchronize()
    print("Average time of FasterGeneration of PLATO-XL model is {}ms. ".format(
        (time.time() - start) / 100 * 1000))

    # TODO: verify post process.
    result = postprocess_response(outputs[0].numpy(), tokenizer)
    result = "".join(result)

    print("Model input:", context)
    print("Result:", result)


if __name__ == "__main__":
    args = setup_args()
    pprint(args)

    infer(args)
