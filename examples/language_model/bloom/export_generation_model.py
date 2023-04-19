# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import argparse
import os

import paddle

from paddlenlp.transformers import AutoTokenizer, BloomConfig, BloomForCausalLM

MODEL_CLASSES = {"bloom": (BloomForCausalLM)}


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_type",
        default="bloom",
        type=str,
        # required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_dtype",
        default="float32",
        type=str,
        help="Model dtype selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bigscience/bloom-560m",
        type=str,
        required=False,
        help="name or path of the trained model to be exported.",
    )
    parser.add_argument(
        "--output_path",
        default="./pretrained/bloom-560m-generation/bloom",
        type=str,
        # required=True,
        help="The output file prefix used to save the exported inference model.",
    )
    parser.add_argument(
        "--max_length",
        default=20,
        type=int,
        help="max length of output sentence",
    )
    args = parser.parse_args()
    return args


def main():
    # most dtype of bloom model weights are float16, expect Bloom(176B) is bfloat16
    args = parse_args()
    paddle.set_default_dtype(args.model_dtype)

    args.model_type = args.model_type.lower()
    model_class = MODEL_CLASSES[args.model_type]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    config = BloomConfig.from_pretrained(args.model_name_or_path)
    config.use_recompute = False
    # Load the model and parameter
    model = model_class.from_pretrained(args.model_name_or_path, config=config, low_cpu_mem_usage=True)

    model.eval()
    input_spec = [
        paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
        None,
        None,
        # max_length
        args.max_length,
        # min_length
        0,
        # decode_strategy
        "sampling",
        # temperature
        1.0,
        # top_k
        1,
        # top_p
        1.0,
        1.0,
        # repetition_penalty
        1,
        # num_beam_groups
        1,
        0.0,
        # early_stopping
        False,
        # bos_token_id
        tokenizer.bos_token_id,
        # eos_token_id
        tokenizer.eos_token_id,
        # pad_token_id
        tokenizer.pad_token_id,
    ]
    model = paddle.jit.to_static(model.generate, input_spec=input_spec)

    # # Save converted static graph model
    paddle.jit.save(model, args.output_path)
    # # Also save tokenizer for inference usage
    tokenizer.save_pretrained(os.path.dirname(args.output_path))


if __name__ == "__main__":
    main()
