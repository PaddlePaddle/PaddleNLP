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
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig


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
    parser.add_argument("--dtype", default=None, help="The data type of exported model")
    parser.add_argument("--tgt_length", type=int, default=100, help="The batch size of data.")
    parser.add_argument("--lora_path", default=None, help="The directory of LoRA parameters. Default to None")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    paddle.seed(100)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.lora_path is not None:
        lora_config = LoRAConfig.from_pretrained(args.lora_path)
        dtype = lora_config.dtype
    elif args.dtype is not None:
        dtype = args.dtype
    else:
        config = LlamaConfig.from_pretrained(args.model_name_or_path)
        dtype = "float16" if config.dtype is None else config.dtype

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        load_state_as_np=True,
        low_cpu_mem_usage=True,
        use_recompute=False,
        use_cache=True,
        dtype=dtype,
    )
    model.config.fp16_opt_level = None  # For dygraph to static only
    if args.lora_path is not None:
        model = LoRAModel.from_pretrained(model, args.lora_path)
    model.eval()
    model = paddle.jit.to_static(
        model.generate,
        input_spec=[
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # attention_mask
            None,  # position_ids
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
            tokenizer.eos_token_id,
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
