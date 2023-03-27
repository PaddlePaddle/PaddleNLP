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
from model_split_merge import merge_model_parallel

from paddlenlp.transformers import AutoTokenizer, BloomConfig, BloomForGeneration

MODEL_CLASSES = {"bloom": (BloomForGeneration)}


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
    paddle.set_default_dtype("float16")
    args = parse_args()

    args.model_type = args.model_type.lower()
    model_class = MODEL_CLASSES[args.model_type]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = BloomConfig.from_pretrained(args.model_name_or_path)

    # Set the generaiton the hyperparameter
    config.max_dec_len = args.max_length
    config.temperature = 0.5
    config.decode_strateg = "sampling"
    config.eos_token_id = tokenizer.eos_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    config.use_cache = True
    config.top_k = 1
    config.use_recompute = False
    config.use_pure_fp16 = False

    args.model_name_or_path = merge_model_parallel(args.model_name_or_path, config)

    # Load the model and parameter
    config.mp_degree = 1
    model = model_class.from_pretrained(args.model_name_or_path, config=config, low_cpu_mem_usage=True)

    model.eval()
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
        ],
    )

    # # Save converted static graph model
    paddle.jit.save(model, args.output_path)
    # # Also save tokenizer for inference usage
    tokenizer.save_pretrained(os.path.dirname(args.output_path))


if __name__ == "__main__":
    main()
