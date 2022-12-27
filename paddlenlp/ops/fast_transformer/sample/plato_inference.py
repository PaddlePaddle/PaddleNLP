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

import numpy as np
import paddle.inference as paddle_infer

from paddlenlp.ops.ext_utils import load
from paddlenlp.transformers import UnifiedTransformerTokenizer


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference_model_dir", default="./infer_model/", type=str, help="Path to save inference model of PLATO. "
    )
    parser.add_argument("--use_role", action="store_true", help="Whether to use role embeddings. ")
    parser.add_argument(
        "--position_style",
        default="relative",
        choices=["continuous", "relative"],
        type=str,
        help="The type for positional embedding. Default is continuous. ",
    )

    args = parser.parse_args()

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
    model_name = "plato-xl"
    tokenizer = UnifiedTransformerTokenizer.from_pretrained(model_name)

    context = [
        "Hi , Becky , what's up ?",
        "Not much , except that my mother-in-law is driving me up the wall .",
        "What's the problem ?",
    ]

    data = tokenizer.dialogue_encode(
        history=context,
        add_start_token_as_response=True,
        return_length=True,
        return_role_ids=args.use_role,
        position_style=args.position_style,
    )

    # Load FastGeneration lib.
    load("FastGeneration", verbose=True)

    config = paddle_infer.Config(
        args.inference_model_dir + "plato.pdmodel", args.inference_model_dir + "plato.pdiparams"
    )
    config.enable_use_gpu(100, 0)
    config.disable_glog_info()
    predictor = paddle_infer.create_predictor(config)

    input_handles = {}
    for name in predictor.get_input_names():
        input_handles[name] = predictor.get_input_handle(name)
        if name == "attention_mask":
            input_handles[name].copy_from_cpu(np.expand_dims(np.asarray(data[name], dtype="float32"), axis=(0, 1)))
        else:
            input_handles[name].copy_from_cpu(np.asarray(data[name], dtype="int32").reshape([1, -1]))

    output_handles = [predictor.get_output_handle(name) for name in predictor.get_output_names()]

    predictor.run()

    output = [output_handle.copy_to_cpu() for output_handle in output_handles]

    for sample in output[0].transpose([1, 0]).tolist():
        print(" ".join(postprocess_response(sample, tokenizer)))


if __name__ == "__main__":
    args = setup_args()
    pprint(args)

    infer(args)
