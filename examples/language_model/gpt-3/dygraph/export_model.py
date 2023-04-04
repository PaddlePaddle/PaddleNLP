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
from modeling import GPTForGeneration
from utils import left_padding, merge_model_parallel

from paddlenlp.transformers import (  # GPTChineseTokenizer,; GPTForGreedyGeneration,
    GPTConfig,
    GPTTokenizer,
)

MODEL_CLASSES = {
    "gpt2": (GPTForGeneration, GPTTokenizer)
    # "gpt2": (GPTLMHeadModel, GPTTokenizer)
}


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_type",
        default="gpt2",
        type=str,
        # required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Path of the trained model to be exported.",
    )
    parser.add_argument(
        "--output_path",
        default="inference/gpt",
        type=str,
        # required=True,
        help="The output file prefix used to save the exported inference model.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Suild model and load trained parameters
    tokenizer = tokenizer_class.from_pretrained(args.model_path)
    # model = model_class.from_config(args.model_path, max_predict_len=32, eol_token_id=tokenizer.eol_token_id)
    # config = GPTConfig.from_pretrained(args.model_path)
    # args.model_path = "gpt2-medium-en"
    config = GPTConfig.from_pretrained(args.model_path)

    config.fuse_attention_qkv = True
    # config.max_predict_len = 8
    config.max_dec_len = 20
    config.eos_token_id = tokenizer.eos_token_id
    config.eol_token_id = tokenizer.eol_token_id
    config.pad_token_id = tokenizer.eos_token_id
    config.use_cache = True
    config.top_k = 1

    model = model_class(config)
    # model = model_class.from_pretrained(args.model_path, config=config)
    missing_keys, unexpected_keys = model.set_state_dict(merge_model_parallel(args.model_path, config))
    print("missing_keys", missing_keys)
    print("unexpected_keys", unexpected_keys)
    # Switch to eval model
    model.eval()
    # Convert to static graph with specific input description
    input_text = ["Nice to meet", "Hello "]
    inputs = tokenizer(input_text)

    # input_ids = tokenizer.encode(input_text)['input_ids']
    inputs = tokenizer(input_text)
    inputs = left_padding(inputs, tokenizer.bos_token_id)
    input_ids = inputs["input_ids"]

    input_ids = paddle.to_tensor(input_ids, dtype="int64")
    ret = model(input_ids=input_ids)

    # ret =  model.generate(input_ids = data["input_ids"])
    for out_ids, in_txt in zip(ret[0].tolist(), input_text):
        print("==" * 30)
        print(in_txt + tokenizer.convert_ids_to_string(out_ids))

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
