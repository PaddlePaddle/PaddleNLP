# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.transformers import T5ForConditionalGeneration, T5Tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", default=256, type=int, help="Maximum output sequence length.")
    parser.add_argument("--beam_size", default=4, type=int, help="The beam size to set.")
    parser.add_argument("--use_faster", action="store_true", help="Whether to use faster to predict.")
    parser.add_argument("--use_fp16_decoding", action="store_true", help="Whether to use fp16 to predict.")
    args = parser.parse_args()
    return args


def predict(args):
    model_name = "t5-base"

    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    en_text = ' This image section from an infrared recording by the Spitzer telescope shows a "family portrait" of countless generations of stars: the oldest stars are seen as blue dots. '
    input_ids = tokenizer.encode("translate English to French: " + en_text, return_tensors="pd")["input_ids"]

    output, _ = model.generate(
        input_ids=input_ids,
        num_beams=args.beam_size,
        max_length=args.max_length,
        decode_strategy="beam_search",
        use_fast=True,  # args.use_faster,
        use_fp16_decoding=args.use_fp16_decoding,
    )

    translation = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    print("The original sentence: ", en_text)
    print("The translation result: ", translation)


if __name__ == "__main__":
    args = parse_args()

    predict(args)
