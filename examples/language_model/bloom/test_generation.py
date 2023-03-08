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
from transformers import AutoTokenizer 
from modeling import BloomForGeneration
from configuration import BloomConfig
from utils import left_padding

MODEL_CLASSES = {
    "bigscience/bloom-560m":(BloomForGeneration)
}


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_type",
        default="bigscience/bloom-560m",
        type=str,
        # required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_path",
        default="bigscience/bloom-560m",
        type=str,
        required=False,
        help="Path of the trained model to be exported.",
    )
    parser.add_argument(
        "--output_path",
        default="inference/bloom",
        type=str,
        # required=True,
        help="The output file prefix used to save the exported inference model.",
    )
    args = parser.parse_args()
    return args




def main():
    args = parse_args()

    args.model_type = args.model_type.lower()
    model_class = MODEL_CLASSES[args.model_type]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = BloomConfig.from_pretrained(args.model_path)

    config.max_dec_len = 20
    config.temperature = 0.5
    config.decode_strateg = 'sampling'
    config.eos_token_id = tokenizer.eos_token_id
    config.bos_token_id  = tokenizer.bos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    config.use_cache = True
    config.top_k = 1
    print("config:{}".format(config))

    model = model_class.from_pretrained(args.model_path, config=config)
    
    model.eval()
    # Convert to static graph with specific input description
    input_text = ["hello world", 'i love you']
    inputs = tokenizer(input_text)
    inputs = left_padding(inputs, tokenizer.pad_token_id)
    input_ids = inputs["input_ids"]

    input_ids = paddle.to_tensor(input_ids, dtype="int64")
    ret = model(input_ids=input_ids)

    # ret =  model.generate(input_ids = data["input_ids"])
    for x in ret[0].tolist():
        print("==" * 30)
        tokens = tokenizer.convert_ids_to_tokens(x)
        sentence = tokenizer.convert_tokens_to_string(tokens)
        print(sentence)

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
