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

from paddlenlp.transformers import MT5ForConditionalGeneration, T5Tokenizer

parser = argparse.ArgumentParser("ERNIE-CODE")
parser.add_argument(
    "--model_name_or_path",
    default="ernie-code-base",
    type=str,
)
parser.add_argument("--input", default="BadZipFileのAliasは、古い Python バージョンとの互換性のために。", type=str)
parser.add_argument("--source_lang", default="text", type=str)
parser.add_argument("--target_lang", default="code", type=str)
parser.add_argument("--source_prefix", default="translate Japanese to Python: \n", type=str)
parser.add_argument("--max_length", type=int, default=1024)
parser.add_argument("--num_beams", type=int, default=3)
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu"])

args = parser.parse_args()


def predict():
    def clean_up_codem_spaces(s: str):
        # post process
        # ===========================
        new_tokens = ["<pad>", "</s>", "<unk>", "\n", "\t", "<|space|>" * 4, "<|space|>" * 2, "<|space|>"]
        for tok in new_tokens:
            s = s.replace(f"{tok} ", tok)

        cleaned_tokens = ["<pad>", "</s>", "<unk>"]
        for tok in cleaned_tokens:
            s = s.replace(tok, "")
        s = s.replace("<|space|>", " ")
        # ===========================
        return s

    def postprocess_text(preds):
        preds = [pred.strip() for pred in preds]
        return preds

    def postprocess_code(preds):
        preds = [clean_up_codem_spaces(pred).strip() for pred in preds]
        return preds

    paddle.set_device(args.device)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    model = MT5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
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
        decoded_preds = postprocess_text(decoded_preds)
    elif args.target_lang == "code":
        decoded_preds = tokenizer.batch_decode(
            generated_tokens.numpy(), skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        decoded_preds = postprocess_code(decoded_preds)
    print(decoded_preds)


if __name__ == "__main__":
    predict()
