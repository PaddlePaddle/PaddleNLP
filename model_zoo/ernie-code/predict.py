#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle

from paddlenlp.transformers import AutoModelForConditionalGeneration, AutoTokenizer

parser = argparse.ArgumentParser("ERNIE-CODE")
parser.add_argument(
    "--model_name_or_path",
    default="ernie-code-base-L512",
    type=str,
)
parser.add_argument("--input", default="BadZipFileのAliasは、古い Python バージョンとの互換性のために。", type=str)
parser.add_argument("--target_lang", default="code", type=str)
parser.add_argument("--source_prefix", default="translate Japanese to Python: \n", type=str)
parser.add_argument("--max_length", type=int, default=1024)
parser.add_argument("--num_beams", type=int, default=3)
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu"])

args = parser.parse_args()


def predict():

    paddle.set_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForConditionalGeneration.from_pretrained(args.model_name_or_path)

    prefix = args.source_prefix if args.source_prefix is not None else ""

    def preprocess_function(inputs, tokenizer):
        inputs = [prefix + inp for inp in inputs]

        model_inputs = tokenizer(inputs, max_length=args.max_length)
        return model_inputs

    dev_dataset = [args.input]
    model_inputs = preprocess_function(dev_dataset, tokenizer)
    model.eval()
    gen_kwargs = {
        "max_length": args.max_length,
        "num_beams": args.num_beams,
        "decode_strategy": "beam_search",
        "length_penalty": 0,
        "min_length": 0,
    }
    generated_tokens, _ = model.generate(
        paddle.to_tensor(np.array(model_inputs["input_ids"]).reshape(1, -1).astype("int64")),
        attention_mask=paddle.to_tensor(np.array(model_inputs["attention_mask"]).reshape(1, -1).astype("int64")),
        **gen_kwargs,
    )
    if args.target_lang == "text":
        decoded_preds = tokenizer.batch_decode(generated_tokens.numpy(), skip_special_tokens=True)
    elif args.target_lang == "code":
        decoded_preds = tokenizer.batch_decode(
            generated_tokens.numpy(), skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    print(decoded_preds)


if __name__ == "__main__":
    predict()
