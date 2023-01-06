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
import os
from multiprocessing import Pool

from nltk import sent_tokenize
from rouge import Rouge
from tqdm import tqdm

from paddlenlp.transformers import PegasusChineseTokenizer

rouge = Rouge()


def compute_rouge(target, pred):
    try:
        score = rouge.get_scores(pred[0], target)
        rouge1, rouge2, rougel = score[0]["rouge-1"]["f"], score[0]["rouge-2"]["f"], score[0]["rouge-l"]["f"]
    except ValueError:
        rouge1, rouge2, rougel = 0, 0, 0

    return (rouge1 + rouge2 + rougel) / 3


def data_reformat(args):
    split = args.split
    tokenizer = PegasusChineseTokenizer.from_pretrained("IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese")
    print("start reformat!")
    print("need file: data/split.json and data_cand/split.out")
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
            if i % 10000 == 0:
                print(i)
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
    print("reformat finish!")
    print("data saved in data_cand/")


def collect_diverse_beam_data(args):
    split = args.split
    src_dir = args.src_dir
    tgt_dir = os.path.join(args.tgt_dir, split)
    cands = []
    cands_untok = []
    cnt = 0
    with open(os.path.join(src_dir, f"{split}.source.tokenized")) as src, open(
        os.path.join(src_dir, f"{split}.target.tokenized")
    ) as tgt, open(os.path.join(src_dir, f"{split}.source")) as src_untok, open(
        os.path.join(src_dir, f"{split}.target")
    ) as tgt_untok:
        with open(os.path.join(src_dir, f"{split}.out.tokenized")) as f_1, open(
            os.path.join(src_dir, f"{split}.out")
        ) as f_2:
            for (x, y) in zip(f_1, f_2):
                x = x.strip()
                if args.lower:
                    x = x.lower()
                cands.append(x)
                y = y.strip()
                if args.lower:
                    y = y.lower()
                cands_untok.append(y)
                if len(cands) == args.cand_num:
                    src_line = src.readline()
                    src_line = src_line.strip()
                    if args.lower:
                        src_line = src_line.lower()
                    tgt_line = tgt.readline()
                    tgt_line = tgt_line.strip()
                    if args.lower:
                        tgt_line = tgt_line.lower()
                    src_line_untok = src_untok.readline()
                    src_line_untok = src_line_untok.strip()
                    if args.lower:
                        src_line_untok = src_line_untok.lower()
                    tgt_line_untok = tgt_untok.readline()
                    tgt_line_untok = tgt_line_untok.strip()
                    if args.lower:
                        tgt_line_untok = tgt_line_untok.lower()
                    yield (
                        src_line,
                        tgt_line,
                        cands,
                        src_line_untok,
                        tgt_line_untok,
                        cands_untok,
                        os.path.join(tgt_dir, f"{cnt}.json"),
                        args.dataset,
                    )
                    cands = []
                    cands_untok = []
                    cnt += 1


def build_diverse_beam(input):
    src_line, tgt_line, cands, src_line_untok, tgt_line_untok, cands_untok, tgt_dir, dataset = input
    cands = [sent_tokenize(x) for x in cands]
    abstract = sent_tokenize(tgt_line)
    _abstract = "\n".join(abstract)
    article = sent_tokenize(src_line)
    if [] in cands:
        not_none_cand = ""
        none_idxs = []
        for idx, item in enumerate(cands):
            if item != [] and not_none_cand == "":
                not_none_cand = item
            if item == []:
                none_idxs.append(idx)
        for i in none_idxs:
            cands[i] = not_none_cand

    candidates = [(x, compute_rouge(_abstract, x)) for x in cands]
    cands_untok = [sent_tokenize(x) for x in cands_untok]
    abstract_untok = sent_tokenize(tgt_line_untok)
    article_untok = sent_tokenize(src_line_untok)
    candidates_untok = [(cands_untok[i], candidates[i][1]) for i in range(len(candidates))]
    output = {
        "article": article,
        "abstract": abstract,
        "candidates": candidates,
        "article_untok": article_untok,
        "abstract_untok": abstract_untok,
        "candidates_untok": candidates_untok,
    }
    with open(tgt_dir, "w") as f:
        json.dump(output, f, ensure_ascii=False)


def make_diverse_beam_data(args):
    print("start make diverse data")
    with open(os.path.join(args.src_dir, f"{args.split}.source")) as f:
        num = sum(1 for _ in f)
    data = collect_diverse_beam_data(args)
    with Pool(processes=8) as pool:
        for _ in tqdm(pool.imap_unordered(build_diverse_beam, data), total=num):
            pass
    print("finish")
    print("data saved in data/diverse/split/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing Parameter")
    parser.add_argument("--cand_num", type=int, default=16, help="Number of candidates")
    parser.add_argument("--src_dir", type=str, default="data_cand/", help="Source directory")
    parser.add_argument("--tgt_dir", type=str, default="data/diverse/", help="Target directory")
    parser.add_argument("--split", type=str, default="test", help="Dataset Split")
    parser.add_argument("--dataset", type=str, default="cnndm", help="Dataset")
    parser.add_argument("-l", "--lower", action="store_true", help="Lowercase")
    args = parser.parse_args()
    data_reformat(args)
    make_diverse_beam_data(args)
