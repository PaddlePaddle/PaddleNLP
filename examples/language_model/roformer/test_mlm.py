# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import argparse
from paddlenlp.transformers import RoFormerForMaskedLM, RoFormerTokenizer


def test_mlm(text, model_name):
    model = RoFormerForMaskedLM.from_pretrained(model_name)
    model.eval()
    tokenizer = RoFormerTokenizer.from_pretrained(model_name)
    tokens = ["[CLS]"]
    text_list = text.split("[MASK]")
    for i, t in enumerate(text_list):
        tokens.extend(tokenizer.tokenize(t))
        if i == len(text_list) - 1:
            tokens.extend(["[SEP]"])
        else:
            tokens.extend(["[MASK]"])

    input_ids_list = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = paddle.to_tensor([input_ids_list])

    with paddle.no_grad():
        pd_outputs = model(input_ids)[0]
    pd_outputs_sentence = "paddle: "
    for i, id in enumerate(input_ids_list):
        if id == tokenizer.convert_tokens_to_ids(["[MASK]"])[0]:
            tokens = tokenizer.convert_ids_to_tokens(
                pd_outputs[i].topk(5)[1].tolist())
            pd_outputs_sentence += "[" + "||".join(tokens) + "]"
        else:
            pd_outputs_sentence += "".join(
                tokenizer.convert_ids_to_tokens([id], skip_special_tokens=True))

    print(pd_outputs_sentence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        default="roformer-chinese-base",
                        type=str,
                        help="Pretrained roformer name or path.")
    parser.add_argument("--text",
                        default="今天[MASK]很好，我想去公园玩！",
                        type=str,
                        help="MLM text.")
    args = parser.parse_args()
    test_mlm(text=args.text, model_name=args.model_name)
