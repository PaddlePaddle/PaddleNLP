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
from paddlenlp.transformers import AutoModelForCausalLM


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, required=True, help="The directory of pretrained model.")
    parser.add_argument(
        "--lora_path", default=None, required=True, help="The directory of LoRA parameters. Default to None"
    )
    parser.add_argument("--merge_model_path", default=None, help="The directory of merged parameters. Default to None")
    parser.add_argument("--use_vocab_extend", default=False, help="Whether to use vocab_extend")
    parser.add_argument("--device", type=str, default="gpu", help="Device")
    return parser.parse_args()


def merge():
    args = parse_arguments()
    paddle.set_device(args.device)
    lora_config = LoRAConfig.from_pretrained(args.lora_path)
    dtype = lora_config.dtype
    lora_config.merge_weights = True

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=dtype,
    )

    if args.use_vocab_extend:
        LORA_FILE_NAME = "lora_model_state.pdparams"
        vocab_extend_state_dict = paddle.load(os.path.join(args.lora_path, LORA_FILE_NAME))
        for key, value in vocab_extend_state_dict.items():
            if "embed_tokens" in key:
                model.resize_token_embeddings(vocab_extend_state_dict[key].shape[0])
                with paddle.no_grad():
                    paddle.set_default_dtype("float16")
                    new_embed_tokens = paddle.nn.Embedding(
                        vocab_extend_state_dict[key].shape[0], vocab_extend_state_dict[key].shape[1]
                    )
                    new_embed_tokens.weight[:, :] = value[:, :]
                    paddle.set_default_dtype("float32")
                model.set_input_embeddings(new_embed_tokens)
            if "lm_head" in key:
                with paddle.no_grad():
                    paddle.set_default_dtype("float16")
                    new_lm_head_weight = paddle.create_parameter(
                        shape=[vocab_extend_state_dict[key].shape[0], vocab_extend_state_dict[key].shape[1]],
                        dtype=paddle.get_default_dtype(),
                    )
                    new_lm_head_weight[:, :] = value[:, :]
                    paddle.set_default_dtype("float32")
                model.lm_head.weight = new_lm_head_weight

    model = LoRAModel.from_pretrained(model=model, lora_path=args.lora_path, lora_config=lora_config)
    model.eval()
    if args.merge_model_path is None:
        args.merge_model_path = args.lora_path

    model_state_dict = model.model.state_dict()
    for key in list(model_state_dict):
        if "lora" in key:
            del model_state_dict[key]
    model.model.save_pretrained(args.merge_model_path, state_dict=model_state_dict)


if __name__ == "__main__":
    merge()
