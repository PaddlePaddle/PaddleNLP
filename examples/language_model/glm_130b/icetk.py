# -*- encoding: utf-8 -*-

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

"""
@File    :   text_tokenizer.py
@Time    :   2021/12/20 01:26:12
@Author  :   Ming Ding
@Contact :   dm18@mails.tsinghua.edu.cn
"""

# here put the import lib
import os
from typing import List

import requests
import sentencepiece as spm
import sentencepiece_model_pb2 as model
from filelock import FileLock
from tqdm import tqdm


class TextTokenizer:
    def __init__(self, model_path):
        self.proto = model.ModelProto()
        with open(model_path, "rb") as fin:
            proto_str = fin.read()
            self.proto.ParseFromString(proto_str)
        self.refresh()

    def refresh(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_proto=self.proto.SerializeToString())
        self.num_tokens = self.sp.vocab_size()

    def add_special_tokens(self, tokens):
        for token in tokens:
            new_token = model.ModelProto().SentencePiece()
            new_token.piece = token
            new_token.score = 0
            self.proto.pieces.append(new_token)
        self.refresh()

    def discourage_tokens(self, tokens):
        if isinstance(tokens, str):  # single token
            tokens = [tokens]
        for token in tokens:
            for piece in self.proto.pieces:
                if piece.piece == token:
                    piece.score = -100
        self.refresh()

    def discourage_ids(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        for idx in ids:
            self.proto.pieces[idx].score = -100
        self.refresh()

    def encode(self, text):
        return self.sp.EncodeAsIds(text)

    def decode(self, ids: List[int]):
        return self.sp.DecodeIds(ids)

    def tokenize(self, text):
        return self.sp.EncodeAsPieces(text)

    def convert_tokens_to_ids(self, tokens):
        return [self.sp.PieceToId(token) for token in tokens]

    def convert_token_to_id(self, token):
        return self.sp.PieceToId(token)

    def convert_id_to_token(self, idx):
        return self.sp.IdToPiece(idx)

    def __len__(self):
        return self.num_tokens


MODEL_ULRS = {
    "ice_text.model": "https://cloud.tsinghua.edu.cn/f/2c73ea6d3e7f4aed82ec/?dl=1",
    "ice_image.pt": "https://cloud.tsinghua.edu.cn/f/ae2cd37af814429d875d/?dl=1",
}


def download_with_progress_bar(save_path, url):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pbar = tqdm(total=int(r.headers["Content-Length"]), unit_scale=True)
            for chunk in r.iter_content(chunk_size=32 * 1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))


def auto_create(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    lock = FileLock(file_path + ".lock")
    with lock:
        if os.path.exists(file_path):
            return False
        else:
            url = MODEL_ULRS[os.path.basename(file_path)]
            print(f"Downloading tokenizer models {url} into {file_path} ...")
            download_with_progress_bar(file_path, url)
            return True
