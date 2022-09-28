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
from paddlenlp.transformers import ReformerModelWithLMHead


# Encoding
def encode(list_of_strings, pad_token_id=0):
    max_length = max([len(string) for string in list_of_strings])

    # create emtpy tensors
    attention_masks = paddle.zeros((len(list_of_strings), max_length),
                                   dtype="int64")
    input_ids = paddle.full((len(list_of_strings), max_length),
                            pad_token_id,
                            dtype="int64")

    for idx, string in enumerate(list_of_strings):
        # make sure string is in byte format
        if not isinstance(string, bytes):
            string = str.encode(string)

        input_ids[idx, :len(string)] = paddle.to_tensor([x + 2 for x in string],
                                                        dtype="int64")
        attention_masks[idx, :len(string)] = 1

    return input_ids, attention_masks


# Decoding
def decode(outputs_ids):
    decoded_outputs = []
    for output_ids in outputs_ids.tolist():
        # transform id back to char IDs < 2 are simply transformed to ""
        decoded_outputs.append("".join(
            [chr(x - 2) if x > 1 else "" for x in output_ids]))
    return decoded_outputs


if __name__ == "__main__":
    model = ReformerModelWithLMHead.from_pretrained("reformer-enwik8")
    model.eval()
    encoded, attention_masks = encode(
        ["In 1965, Brooks left IBM to found the Department of"])
    output = decode(
        model.generate(encoded,
                       decode_strategy='greedy_search',
                       max_length=150,
                       repetition_penalty=1.2)[0])
    print(output)
    # expected:
    # [" Defense. The Department was able to convince the Department to resign from the Department's constitutional amendments to the Department of Defense.\n\n"]
