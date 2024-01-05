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

from dataclasses import dataclass, field

import numpy as np
import paddle
from paddlenlp_ops import get_output

from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import AutoTokenizer, LlamaTokenizer


@dataclass
class ResReaderArgument:
    model_name_or_path: str = field(default=None, metadata={"help": "The directory of model."})


parser = PdArgumentParser((ResReaderArgument))
res_args = parser.parse_args_into_dataclasses()[0]

tokenizer = AutoTokenizer.from_pretrained(res_args.model_name_or_path)
if isinstance(tokenizer, LlamaTokenizer) and not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.unk_token
rank_id = 0
is_blocking = True
paddle.device.set_device("cpu")


output_tensor = paddle.full(shape=[256 + 2, 1], fill_value=2, dtype="int64")
while True:
    outputs = []
    while True:
        get_output(output_tensor, rank_id, is_blocking)
        if output_tensor[0, 0] == -2:  # read none
            continue
        # print("bs: ", output_tensor[1, 0].numpy())
        bsz = output_tensor[1, 0].numpy()
        output_numpy = output_tensor[2 : bsz + 2].numpy()
        # print("output: ", output_numpy[:1])
        output_numpy[output_numpy == -1] = 2
        outputs.append(output_numpy)
        if output_tensor[0, 0] == -1:
            break
    output = np.concatenate(outputs, axis=1).tolist()

    for i, seq in enumerate(output):
        print(i, "seq_len: ", len(seq))
        seq = tokenizer.decode(seq)

        print("output: ", repr(seq))
    print("end")
