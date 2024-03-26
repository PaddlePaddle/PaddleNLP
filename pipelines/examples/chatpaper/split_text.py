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

import argparse
import glob
import os
from multiprocessing import Pool

import jsonlines
import pandas as pd
from tqdm import tqdm

from pipelines.nodes import SpacyTextSplitter

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--file_paths", default='data/abstracts', type=str, help="The PDF file path.")
args = parser.parse_args()
# yapf: enable


def read_data(file_path):
    data = pd.read_json(path_or_buf=file_path, lines=True)
    list_data = []
    for index, row in tqdm(data.iterrows()):
        doc = row.to_dict()
        list_data.append(doc)
    return list_data


def indexing_abstract(file_path):
    dataset = read_data(file_path)
    text_splitter = SpacyTextSplitter(separator="\n", chunk_size=420, chunk_overlap=10, filters=["\n"])
    datasets = []
    for document in tqdm(dataset):
        text = document["abstract"]
        text_splits = text_splitter.split_text(text)
        for txt in text_splits:
            meta_data = {}
            meta_data.update(document)
            meta_data.pop("content")
            meta_data.pop("abstract")
            datasets.append({"content": txt, "meta": meta_data})

    file_name = file_path.split("/")[-1]
    file_name = file_name.split(".")[0]

    json_name = f"data/chunks/{file_name}_chunks.jsonl"
    with jsonlines.open(json_name, mode="w") as writer:
        for doc in datasets:
            writer.write(doc)


def extract_all_contents(content):
    text_body = []
    cur_idx = len(content)
    # improve efficiency
    for index in range(len(content) - 1, -1, -1):
        sentence = content[index]
        cur_idx = index
        try:
            if len(sentence.strip()) == 0:
                continue
            # remove references
            elif "参考文献" in sentence:
                text_body = content[: index - 1]
                break
            # remove english sentence, maybe english abstracts, too slow
            # from langdetect import detect
            # res = detect(sentence)
            # if res == "en":
            #     # print(sentence)
            #     continue
            # text_body.append(sentence)
        except Exception as e:
            print(sentence)
            print(e)
    if cur_idx > 0:
        return text_body
    else:
        return content


def run_multi_process_splitter(file_path):
    dataset = read_data(file_path)
    text_splitter = SpacyTextSplitter(separator="\n", chunk_size=420, chunk_overlap=10, filters=["\n"])
    output_data = []
    for document in tqdm(dataset):
        paragraphs = document["content"].split("\n")
        processed_content = extract_all_contents(paragraphs)
        text = "\n".join(processed_content)
        text_splits = text_splitter.split_text(text)
        for txt in text_splits:
            meta_data = {
                "name": document["title"],
                "id": document["id"],
                "title": document["title"],
                "key_words": document["key_words"],
            }
            output_data.append({"content": txt, "meta": meta_data})
    file_name = file_path.split("/")[-1]
    file_name = file_name.split(".")[0]
    json_name = f"data/full_text/{file_name}_full_chunks.jsonl"
    with jsonlines.open(json_name, mode="w") as writer:
        for doc in output_data:
            writer.write(doc)


def split_abstract(file_paths):
    pool = Pool(processes=64)
    pool.map(indexing_abstract, file_paths)


def split_full_text(file_paths):
    pool = Pool(processes=80)
    pool.map(run_multi_process_splitter, file_paths)
    pool.close()  # close the process pool and no longer accept new processes
    pool.join()


if __name__ == "__main__":
    root_path = args.file_paths
    file_paths = glob.glob(root_path + "/*.jsonl")
    # Split abstract
    os.makedirs("data/chunks", exist_ok=True)
    split_abstract(file_paths)
    # Split full text
    os.makedirs("data/full_text", exist_ok=True)
    split_full_text(file_paths)
