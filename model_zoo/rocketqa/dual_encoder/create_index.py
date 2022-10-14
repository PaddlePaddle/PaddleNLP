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

import sys
import os
import argparse
import time
import math
import glob

import faiss
import numpy as np
from tqdm import tqdm

# yapf: disable
parser = argparse.ArgumentParser()

parser.add_argument("--embedding_path",
                        type=str,
                        default='output',
                        help="The paragraph embedding path.")
parser.add_argument("--save_path",
                        type=str,
                        default='output',
                        help="The paragraph embedding save path.")
parser.add_argument("--embedding_dim", default=768, type=int, help="The embedding dim for indexing.")
parser.add_argument("--output_index_path", default='para.index.part', type=str, help="The output path for indexing.")
args = parser.parse_args()
# yapf: enable


def build_engine(para_emb_list, dim):
    index = faiss.IndexFlatIP(dim)
    # add paragraph embedding
    p_emb_matrix = np.asarray(para_emb_list)
    index.add(p_emb_matrix.astype('float32'))
    return index


def build_index(file_path):
    # for idx,item in tqdm(enumerate(file_paths)):
    #     para_embs = np.load(item)
    #     engine = build_engine(para_embs, args.embedding_dim)
    #     output_file_name = os.path.join(args.embedding_path,'para.index.part{}'.format(idx))
    #     faiss.write_index(engine, output_file_name)
    para_embs = np.load(file_path)
    engine = build_engine(para_embs, args.embedding_dim)
    output_file_name = os.path.join(args.save_path, args.output_index_path)
    faiss.write_index(engine, output_file_name)


def main():
    # file_paths = ['output/part-00.npy','output/part-01.npy','output/part-02.npy','output/part-03.npy']
    build_index(args.embedding_path)


if __name__ == "__main__":
    main()
