# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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

import os
import re
import argparse
import json
import multiprocessing
import sys
import time
import shutil
from functools import partial

import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
                        type=str,
                        required=True,
                        help='Path to you raw files. Folder or file path.')
    parser.add_argument('--output_path',
                        type=str,
                        required=True,
                        help='Path to save the output json files.')
    parser.add_argument('--json_key',
                        type=str,
                        default='text',
                        help='The content key of json file.')
    parser.add_argument(
        '--doc_spliter',
        type=str,
        default='',
        help=
        "Spliter between documents. We will strip the line, if you use blank line to split doc, leave it blank."
    )
    parser.add_argument('--min_doc_length',
                        type=int,
                        default=10,
                        help="Minimal char of a documment.")
    parser.add_argument('--workers',
                        type=int,
                        default=1,
                        help='Number of worker processes to launch')
    parser.add_argument('--log_interval',
                        type=int,
                        default=1,
                        help='Interval between progress updates.')
    parser.add_argument('--no-merge',
                        action='store_true',
                        help='Don\'t merge the file.')
    parser.add_argument('--no-shuffle',
                        action='store_true',
                        help='Don\'t shuffle the file.')
    args = parser.parse_args()
    return args


def raw_text_to_json(path, doc_spliter="", json_key="text", min_doc_length=10):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        print("No found file %s" % path)
        return 0, None

    out_filepath = path + ".jsonl"
    fout = open(out_filepath, "w", encoding="utf-8")
    len_files = 0
    with open(path, "r") as f:
        doc = ""
        line = f.readline()
        while line:
            len_files += len(line)
            if line.strip() == doc_spliter:
                if len(doc) > min_doc_length:
                    fout.write(
                        json.dumps({json_key: doc}, ensure_ascii=False) + "\n")
                doc = ""
            else:
                doc += line
            line = f.readline()

        if len(doc) > min_doc_length:
            fout.write(json.dumps({json_key: doc}, ensure_ascii=False) + "\n")
        doc = ""

    return len_files, out_filepath


def merge_file(file_paths, output_path):
    if not output_path.endswith(".jsonl"):
        output_path = output_path + ".jsonl"
    print("Merging files into %s" % output_path)
    with open(output_path, 'wb') as wfd:
        for f in file_paths:
            if f is not None and os.path.exists(f):
                with open(f, 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)
                os.remove(f)
    print("File save in %s" % output_path)
    return output_path


def shuffle_file(output_path):
    print("Shuffling the jsonl file...")
    if os.path.exists(output_path):
        os.system("shuf %s -o %s" % (output_path, output_path))
        print("File shuffled!!!")
    else:
        raise ValueError("File not found: %s" % output_path)


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

    trans_json = partial(raw_text_to_json,
                         doc_spliter=args.doc_spliter,
                         json_key=args.json_key,
                         min_doc_length=args.min_doc_length)
    encoded_files = pool.imap(trans_json, file_paths, 1)

    out_paths = []
    for i, (bytes_processed, out_path) in enumerate(encoded_files, start=1):
        total_bytes_processed += bytes_processed
        out_paths.append(out_path)
        master_start = time.time()

        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {i} files",
                  f"({i/elapsed} files/s, {mbs} MB/s).",
                  file=sys.stderr)

    if not args.no_merge:
        output_path = merge_file(out_paths, args.output_path)
        if not args.no_shuffle:
            shuffle_file(output_path)


if __name__ == "__main__":
    main()
    #profile.run("main()", "testprof")
