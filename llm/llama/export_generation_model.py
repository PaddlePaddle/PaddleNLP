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
        default="ziqingyang/chinese-alpaca-7b",
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
    parser.add_argument("--dtype", default="float16", help="The data type of exported model")
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
        config = LlamaConfig.from_pretrained(args.model_path)
        dtype = "float16" if config.dtype is None else config.dtype

    paddle.set_default_dtype(dtype)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        load_state_as_np=True,
        low_cpu_mem_usage=True,
        use_cache=True,
        dtype=dtype,
    )

    if args.lora_path is not None:
        model = LoRAModel.from_pretrained(model, args.lora_path)

    model.prepare_fast_entry({})
    config = {
        "use_top_p": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "use_pre_caches": False,
        "num_layers": model.config.num_hidden_layers,
    }
    model.to_static(args.output_path, config)
    model.config.save_pretrained("inference")
    tokenizer.save_pretrained(os.path.dirname(args.output_path))


if __name__ == "__main__":
    main()
