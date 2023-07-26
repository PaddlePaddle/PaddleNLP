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
from configuration import GPTConfig
from modeling import GPTForCausalLM

from paddlenlp.transformers import GPTChineseTokenizer, GPTTokenizer

MODEL_CLASSES = {
    "gpt2": (GPTForCausalLM, GPTTokenizer),
    "gpt2-cn": (GPTForCausalLM, GPTChineseTokenizer),
}


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_type",
        default="gpt2-cn",
        type=str,
        # required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_path",
        default="gpt-cpm-large-cn",
        type=str,
        help="Path of the trained model to be exported.",
    )
    parser.add_argument(
        "--output_path",
        default="inference/gpt",
        type=str,
        # required=True,
        help="The output file prefix used to save the exported inference model.",
    )
    parser.add_argument("--tgt_length", type=int, default=100, help="The batch size of data.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    paddle.seed(100)
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_path)
    config = GPTConfig.from_pretrained(args.model_path)
    dtype = config.dtype if config.dtype is not None else "float16"

    model = GPTForCausalLM.from_pretrained(
        args.model_path,
        load_state_as_np=True,
        low_cpu_mem_usage=True,
        dtype=dtype,
        tensor_parallel_degree=1,
    )
    # TODO(wawltor) Maybe the config pad_token_id is not right?
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = model.config.pad_token_id
    model.eval()
    model = paddle.jit.to_static(
        model.generate,
        input_spec=[
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            None,
            None,
            # paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # position_ids
            args.tgt_length,  # max length
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
            tokenizer.bos_token_id,
            # eos_token_id
            tokenizer.eol_token_id,
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
        ],
    )

    # Save converted static graph model
    paddle.jit.save(model, args.output_path)
    # Also save tokenizer for inference usage
    tokenizer.save_pretrained(os.path.dirname(args.output_path))


if __name__ == "__main__":
    main()
