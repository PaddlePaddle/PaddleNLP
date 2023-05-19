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
import multiprocessing
import os
import re
import sys
import time
from functools import partial


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to you raw files. Folder or file path.")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes to launch")
    parser.add_argument("--output_path", type=str, default="./tmp", help="Path to save the output json files.")
    parser.add_argument(
        "--data_format",
        type=str,
        default="jsonl",
        choices=["jsonl", "wudao"],
        help="Path to you raw files. Folder or file path.",
    )
    parser.add_argument(
        "--cn_seg_func",
        type=str,
        default="jieba",
        choices=["lac", "seg", "jieba"],
        help="Words segment function for chinese words.",
    )
    parser.add_argument("--log_interval", type=int, default=1, help="Interval between progress updates.")
    args = parser.parse_args()
    return args


def lexical_analysis_fn():
    from LAC import LAC

    lac = LAC(mode="lac")

    def process(line):
        words, _ = lac.run(line)
        return words

    return process


def chinese_segmentation_fn():
    from LAC import LAC

    lac_cws = LAC(mode="seg")

    def process(line):
        words = lac_cws.run(line)
        return words

    return process


def jieba_segmentation_fn():
    import jieba

    def process(line):
        words = jieba.cut(line)
        return list(words)

    return process


CHINESE_SEG_FUNC = {
    "lac": lexical_analysis_fn(),
    "seg": chinese_segmentation_fn(),
    "jieba": jieba_segmentation_fn(),
}


def read_wudao(path):
    print("Loading %s" % path)
    with open(path, "r") as f:
        try:
            contents = json.load(f)
        except Exception:
            print("Failed to load %s" % path)
            raise StopIteration
    for js in contents:
        yield js["content"]


def read_jsonl(path):
    print("Loading %s" % path)
    with open(path, "r") as f:
        line = f.readline()
        while line:
            contents = json.load(f)
            yield contents["text"]
            line = f.readline()


READFILE_FUNC = {
    "jsonl": read_jsonl,
    "wudao": read_wudao,
}

special_chars = ["\n", "。", "?", "？", " ", ";", "；", "！", "!"]
split_chars = ["。", "?", "？", ";", "；", "!", "！"]


def text_to_text(path, output_path, read_func, seg_func):
    out_name = os.path.join(output_path, path[-20:])

    print("Write into %s" % out_name)
    if os.path.exists(out_name):
        print("File exists %s" % out_name)
        return 0, None

    seg_func = CHINESE_SEG_FUNC[seg_func]
    read_func = READFILE_FUNC[read_func]

    data_len = 0
    count = 0
    with open(out_name, "w") as f:
        for text in read_func(path):
            # for js in contents:
            count += 1
            # text = js["content"]
            data_len += len(text.encode("utf-8"))
            # make special char only once,
            # because of those token will be treat as sentence spliter.
            # 此处为断句逻辑
            for char in special_chars:
                text = re.sub("[" + char + "]+[ ]*", char, text)
            for char in split_chars:
                text = text.replace(char, char + "\n")

            # 此处为分词逻辑
            final = ""
            for line in text.split("\n"):
                if len(line) == 0:
                    continue
                words = seg_func(line)
                final += " ".join(words) + "\n"
            f.write(final + "\n")

    return data_len, None


def main():
    args = get_args()
    startup_start = time.time()

    file_paths = []
    if os.path.isfile(args.input_path):
        file_paths.append(args.input_path)
    else:
        for root, _, fs in os.walk(args.input_path):
            for f in fs:
                file_paths.append(os.path.join(root, f))

    pool = multiprocessing.Pool(args.workers)

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    trans_func = partial(
        text_to_text, output_path=args.output_path, seg_func=args.cn_seg_func, read_func=args.data_format
    )

    encoded_files = pool.imap(trans_func, file_paths, 1)

    out_paths = []
    for i, (bytes_processed, out_path) in enumerate(encoded_files, start=1):
        total_bytes_processed += bytes_processed
        out_paths.append(out_path)

        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {i} files", f"({i/elapsed} files/s, {mbs} MB/s).", file=sys.stderr)
    pool.close()


if __name__ == "__main__":
    main()
