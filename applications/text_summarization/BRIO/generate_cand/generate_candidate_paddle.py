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
import json
from typing import List

import paddle

from paddlenlp.transformers import (
    PegasusChineseTokenizer,
    PegasusForConditionalGeneration,
)


@paddle.no_grad()
def generate_summaries(args):
    device = f"gpu:{args.gpuid}"
    paddle.set_device(device)
    model_name_or_path = "IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese"
    model = PegasusForConditionalGeneration.from_pretrained(model_name_or_path)
    tokenizer = PegasusChineseTokenizer.from_pretrained(model_name_or_path)
    model.eval()
    count = 0
    bsz = 16
    with open(args.src_dir) as source, open(args.tgt_dir, "w") as fout:
        data = source.readlines()
        slines = []
        for i, sline in enumerate(data):
            sline = json.loads(sline)["content"]
            if count % 1000 == 0:
                print(count)
            if count % bsz == 0 and count > 0:
                batch = tokenizer(slines, padding=True, truncation=True, return_tensors="pd")
                gen, _ = model.generate(
                    **batch,
                    num_return_sequences=128,
                    num_beam_groups=16,
                    diversity_penalty=0.1,
                    num_beams=128,
                    length_penalty=0.6,
                    decode_strategy="beam_search",
                )
                dec: List[str] = tokenizer.batch_decode(gen.numpy(), skip_special_tokens=True)
                dec = [dec[i] for i in range(len(dec)) if i % 8 == 0]
                for hypothesis in dec:
                    fout.write(hypothesis + "\n")
                    fout.flush()
                slines = []
            if len(sline) == 0:
                sline = " "
            slines.append(sline)
            count += 1
        if slines != []:
            batch = tokenizer(slines, padding=True, truncation=True, return_tensors="pd")
            gen, _ = model.generate(
                **batch,
                num_return_sequences=128,
                num_beam_groups=16,
                diversity_penalty=0.1,
                num_beams=128,
                length_penalty=0.6,
                decode_strategy="beam_search",
            )
            dec: List[str] = tokenizer.batch_decode(gen.numpy(), skip_special_tokens=True)
            dec = [dec[i] for i in range(len(dec)) if i % 8 == 0]
            for hypothesis in dec:
                fout.write(hypothesis + "\n")
                fout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument("--gpuid", type=int, default=7, help="gpu id")
    parser.add_argument("--src_dir", type=str, default="data/train.json", help="source file")
    parser.add_argument("--tgt_dir", type=str, default="data_cand/train.out", help="target file")
    args = parser.parse_args()

    generate_summaries(args)
