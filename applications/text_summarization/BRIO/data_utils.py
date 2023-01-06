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
import os

import paddle
from paddle.io import Dataset

from paddlenlp.transformers import PegasusChineseTokenizer


class BrioDataset(Dataset):
    def __init__(
        self,
        fdir,
        model_name_or_path,
        max_len=-1,
        is_test=False,
        total_len=512,
        is_sorted=True,
        max_num=-1,
        is_untok=True,
        num=-1,
    ):
        """data format: article, abstract, [(candidiate_i, score_i)]"""
        self.isdir = os.path.isdir(fdir)
        if self.isdir:
            self.fdir = fdir
            if num > 0:
                self.num = min(len(os.listdir(fdir)), num)
            else:
                self.num = len(os.listdir(fdir))
        else:
            with open(fdir) as f:
                self.files = [x.strip() for x in f]
            if num > 0:
                self.num = min(len(self.files), num)
            else:
                self.num = len(self.files)

        self.tok = PegasusChineseTokenizer.from_pretrained(model_name_or_path, verbose=False)
        self.maxlen = max_len
        self.is_test = is_test
        self.total_len = total_len
        self.sorted = is_sorted
        self.maxnum = max_num
        self.is_untok = is_untok

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if self.isdir:
            with open(os.path.join(self.fdir, "%d.json" % idx), "r") as f:
                data = json.load(f)
        else:
            with open(self.files[idx]) as f:
                data = json.load(f)
        if self.is_untok:
            article = data["article_untok"]
        else:
            article = data["article"]
        src_txt = " ".join(article)
        src = self.tok(
            [src_txt], max_length=self.total_len, return_tensors="pd", pad_to_max_length=False, truncation=True
        )
        src_input_ids = src["input_ids"]
        src_input_ids = src_input_ids.squeeze(0)
        if self.is_untok:
            abstract = data["abstract_untok"]
        else:
            abstract = data["abstract"]
        if self.maxnum > 0:
            candidates = data["candidates_untok"][: self.maxnum]
            _candidates = data["candidates"][: self.maxnum]
            data["candidates"] = _candidates
        if self.sorted:
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            _candidates = sorted(_candidates, key=lambda x: x[1], reverse=True)
            data["candidates"] = _candidates
        if not self.is_untok:
            candidates = _candidates
        cand_txt = [" ".join(abstract)] + [" ".join(x[0]) for x in candidates]
        cand = self.tok(
            cand_txt,
            max_length=self.maxlen,
            return_tensors="pd",
            pad_to_max_length=False,
            truncation=True,
            padding=True,
        )
        candidate_ids = cand["input_ids"]
        _candidate_ids = paddle.zeros(
            shape=[candidate_ids.shape[0], candidate_ids.shape[1] + 1], dtype=candidate_ids.dtype
        )
        _candidate_ids[:, 1:] = paddle.clone(candidate_ids)
        _candidate_ids[:, 0] = self.tok.pad_token_id
        candidate_ids = _candidate_ids

        result = {
            "src_input_ids": src_input_ids,
            "candidate_ids": candidate_ids,
        }
        if self.is_test:
            result["data"] = data
        return result


def collate_mp_brio(batch, pad_token_id, is_test=False):
    def pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(x.shape[0] for x in X)
        result = paddle.ones([len(X), max_len], dtype=X[0].dtype) * pad_token_id
        for (i, x) in enumerate(X):
            target_len = min(max_len, x.shape[0])
            result[i, :target_len] = x[:target_len]
        return result

    src_input_ids = pad([x["src_input_ids"] for x in batch])
    candidate_ids = [x["candidate_ids"] for x in batch]
    max_len = min(24, max([max([len(c) for c in x]) for x in candidate_ids]))
    candidate_ids = [pad(x, max_len) for x in candidate_ids]
    candidate_ids = paddle.stack(candidate_ids)
    if is_test:
        data = [x["data"] for x in batch]
    result = {
        "src_input_ids": src_input_ids,
        "candidate_ids": candidate_ids,
    }
    if is_test:
        result["data"] = data
    return result
