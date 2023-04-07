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

import argparse
import os

import paddle

from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="checkpoints",
        type=str,
        required=False,
        help="Path of the trained model to be exported.",
    )
    parser.add_argument(
        "--output_path",
        default="inference/llama",
        type=str,
        help="The output file prefix used to save the exported inference model.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    paddle.seed(100)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, add_bos_token=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        load_state_as_np=True,
        low_cpu_mem_usage=True,
        use_recompute=False,
        use_cache=True,
    )

    model.eval()
    model = paddle.jit.to_static(
        model.generate,
        input_spec=[
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # attention_mask
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # position_ids
            100,  # max length
            0,  # min length
            "sampling",  # decode_strategy
            1.0,  # temperature
            1,  # top_k
            1.0,  # top_p
            1.0,  # repetition_penalty,
        ],
    )

    # Save converted static graph model
    paddle.jit.save(model, args.output_path)
    # Also save tokenizer for inference usage
    tokenizer.save_pretrained(os.path.dirname(args.output_path))


if __name__ == "__main__":
    main()
