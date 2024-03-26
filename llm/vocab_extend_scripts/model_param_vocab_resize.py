# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="", required=True, help="The directory of pretrained model.")
    parser.add_argument(
        "--new_tokenizer_name_or_path", default="", required=True, help="The directory of new_tokenizer"
    )
    parser.add_argument(
        "--vocab_extend_model_path", default="", required=True, help="path tot save vocab_extend_model"
    )
    parser.add_argument("--device", type=str, default="gpu", help="Device")
    return parser.parse_args()


def main():
    args = parse_arguments()
    paddle.set_device(args.device)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, dtype="float16")
    new_tokenizer = AutoTokenizer.from_pretrained(args.new_tokenizer_name_or_path)

    new_num_tokens = len(new_tokenizer)
    model.base_model.config["vocab_size"] = new_num_tokens
    model.vocab_size = new_num_tokens
    old_token_embeddings = model.get_input_embeddings()
    old_output_embeddings = model.get_output_embeddings()
    new_token_embeddings = paddle.nn.Embedding(new_num_tokens, old_token_embeddings.weight.shape[1])
    if new_token_embeddings.weight.dtype != old_token_embeddings.weight.dtype:
        new_token_embeddings.to(dtype=old_token_embeddings.weight.dtype)
    new_lm_head_weight = paddle.create_parameter(
        shape=[old_output_embeddings.weight.shape[0], new_num_tokens],
        dtype=old_output_embeddings.weight.dtype,
    )
    with paddle.no_grad():
        num_to_copy = min(old_token_embeddings.weight.shape[0], new_num_tokens)
        new_token_embeddings.weight[:num_to_copy, :] = old_token_embeddings.weight[:num_to_copy, :]
        new_lm_head_weight[:, :num_to_copy] = old_output_embeddings.weight[:, :num_to_copy]
    old_output_embeddings.weight = new_lm_head_weight
    model.set_output_embeddings(old_output_embeddings)
    model.set_input_embeddings(new_token_embeddings)

    print(model.lm_head.weight.shape)
    print(model.get_input_embeddings().weight.shape)
    new_tokenizer.save_pretrained(args.vocab_extend_model_path)
    model.save_pretrained(args.vocab_extend_model_path)


if __name__ == "__main__":
    main()
