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

from paddlenlp.peft import LoRAConfig, LoRAModel
from paddlenlp.transformers import ChatGLMConfig, ChatGLMForCausalLM, ChatGLMTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name_or_path",
        default="THUDM/chatglm-6b",
        type=str,
        # required=True,
        help="Path of the trained model to be exported.",
    )
    parser.add_argument(
        "--output_path",
        default="inference/chatglm",
        type=str,
        # required=True,
        help="The output file prefix used to save the exported inference model.",
    )
    parser.add_argument("--dtype", default=None, help="The data type of exported model")
    parser.add_argument("--lora_path", default=None, help="The directory of LoRA parameters. Default to None")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    tokenizer = ChatGLMTokenizer.from_pretrained(args.model_name_or_path)
    if args.lora_path is not None:
        lora_config = LoRAConfig.from_pretrained(args.lora_path)
        dtype = lora_config.dtype
    elif args.dtype is not None:
        dtype = args.dtype
    else:
        config = ChatGLMConfig.from_pretrained(args.model_name_or_path)
        dtype = config.dtype if config.dtype is not None else config.paddle_dtype

    model = ChatGLMForCausalLM.from_pretrained(args.model_name_or_path, load_state_as_np=True, dtype=dtype)
    if args.lora_path is not None:
        model = LoRAModel.from_pretrained(model, args.lora_path)

    model.eval()
    input_spec = [
        paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
        paddle.static.InputSpec(shape=[None, None, None, None], dtype="int64"),  # attention_mask
        paddle.static.InputSpec(shape=[None, None, None], dtype="int64"),  # position_ids
        # max_length
        128,
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
        # repetition_penalty
        1,
        # num_beams
        1,
        # num_beam_groups
        1,
        # length_penalty
        0.0,
        # early_stopping
        False,
        # bos_token_id
        tokenizer.eos_token_id,
        # eos_token_id
        tokenizer.end_token_id,
        # pad_token_id
        tokenizer.pad_token_id,
        # decoder_start_token_id
        None,
        # forced_bos_token_id
        None,
        # forced_eos_token_id
        None,
        # no_repeat_ngram_size
        None,
        # num_return_sequences
        1,
        # diversity_rate
        0.0,
        # use_cache
        True,
    ]
    model = paddle.jit.to_static(model.generate, input_spec=input_spec)

    # # Save converted static graph model
    paddle.jit.save(model, args.output_path)
    # # Also save tokenizer for inference usage
    tokenizer.save_pretrained(os.path.dirname(args.output_path))


if __name__ == "__main__":
    main()
