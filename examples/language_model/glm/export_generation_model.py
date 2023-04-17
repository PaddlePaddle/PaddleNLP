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

from paddlenlp.transformers import AutoModelForConditionalGeneration, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name_or_path",
        default="THUDM/glm-large-chinese",
        type=str,
        # required=True,
        help="Path of the trained model to be exported.",
    )
    parser.add_argument(
        "--output_path",
        default="inference/glm",
        type=str,
        # required=True,
        help="The output file prefix used to save the exported inference model.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForConditionalGeneration.from_pretrained(args.model_name_or_path, load_state_as_np=True)

    model.eval()
    input_spec = [
        paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
        paddle.static.InputSpec(shape=[None, None, None, None], dtype="int64"),
        paddle.static.InputSpec(shape=[None, None, None], dtype="int64"),  # pos_ids
        # max_length
        20,
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
        tokenizer.sop_token_id,
        # eos_token_id
        tokenizer.eop_token_id,
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
