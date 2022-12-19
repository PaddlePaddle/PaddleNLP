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

import json

from paddlenlp.transformers import PegasusChineseTokenizer

num = 500
split = "train"
tokenizer = PegasusChineseTokenizer.from_pretrained("IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese")
with open(f"data/{split}.json") as fin, open(f"data_cand/{split}.source", "w") as fout1, open(
    f"data_cand/{split}.target", "w"
) as fout2, open(f"data_cand/{split}.source.tokenized", "w") as fout3, open(
    f"data_cand/{split}.target.tokenized", "w"
) as fout4:
    datas = fin.readlines()
    source = []
    source_tok = []
    target = []
    target_tok = []
    for i, line in enumerate(datas):
        if i > num:
            break
        data = json.loads(line)
        source.append(data["content"])
        source_tok.append(tokenizer.tokenize(data["content"]))
        target.append(data["title"])
        target_tok.append(tokenizer.tokenize(data["title"]))

    for src in source:
        fout1.write(src + "\n")
        fout1.flush()

    for tgt in target:
        fout2.write(tgt + "\n")
        fout2.flush()

    for src_tok in source_tok:
        fout3.write(" ".join(src_tok) + "\n")
        fout3.flush()

    for tgt_tok in target_tok:
        fout4.write(" ".join(tgt_tok) + "\n")
        fout4.flush()

with open(f"data_cand/{split}.out") as i, open(f"data_cand/{split}.out.tokenized", "w") as o:
    lines = i.readlines()
    out_tok = []
    for line in lines:
        out_tok.append(tokenizer.tokenize(line))
    for o_tok in out_tok:
        o.write(" ".join(o_tok) + "\n")
        o.flush()
