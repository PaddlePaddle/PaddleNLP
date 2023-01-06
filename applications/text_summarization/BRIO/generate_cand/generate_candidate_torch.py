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

import torch
from transformers import PegasusForConditionalGeneration

from paddlenlp.transformers import PegasusChineseTokenizer


def generate_summaries(args):
    device = f"cuda:{args.gpuid}"
    mname = "IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese"
    model = PegasusForConditionalGeneration.from_pretrained(mname)
    model = model.to(device)
    model.eval()
    tok = PegasusChineseTokenizer.from_pretrained(mname)
    bsz = 16
    with open(args.src_dir) as source, open(args.tgt_dir, "w") as fout:
        data = source.readlines()
        slines = []
        for i, sline in enumerate(data):
            sline = json.loads(sline)["content"]
            if i % 1000 == 0:
                print(i)
            if i % bsz == 0 and i > 0:
                with torch.no_grad():
                    batch = tok.prepare_seq2seq_batch(src_texts=slines, return_tensors="pt").to(device)
                    gen = model.generate(
                        **batch,
                        num_return_sequences=16,
                        num_beam_groups=16,
                        diversity_penalty=0.4,
                        num_beams=16,
                        length_penalty=0.6,
                    )
                    dec: List[str] = tok.batch_decode(gen, skip_special_tokens=True)
                # dec = [dec[i] for i in range(len(dec)) if i % 8 == 0]
                for hypothesis in dec:
                    fout.write(hypothesis + "\n")
                    fout.flush()
                slines = []
            sline = sline.strip()
            if len(sline) == 0:
                sline = " "
            slines.append(sline)
        if slines != []:
            with torch.no_grad():
                batch = tok.prepare_seq2seq_batch(src_texts=slines, return_tensors="pt").to(device)
                gen = model.generate(
                    **batch,
                    num_return_sequences=16,
                    num_beam_groups=16,
                    diversity_penalty=0.4,
                    num_beams=16,
                    length_penalty=0.6,
                )
                dec: List[str] = tok.batch_decode(gen, skip_special_tokens=True)
            # dec = [dec[i] for i in range(len(dec)) if i % 8 == 0]
            for hypothesis in dec:
                fout.write(hypothesis + "\n")
                fout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument("--gpuid", type=int, default=0, help="gpu id")
    parser.add_argument("--src_dir", type=str, default="../data/train.json", help="source file")
    parser.add_argument("--tgt_dir", type=str, default="../data_cand/train.out", help="target file")
    args = parser.parse_args()

    generate_summaries(args)
