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
import os
from datetime import datetime

from paddlenlp.data import indexed_dataset


def print_datetime(string):
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[" + string + "] datetime: {} ".format(time_str))


def main(args):

    prefixes = set()
    for basename in os.listdir(args.input):
        prefix, ext = os.path.splitext(basename)

        if prefix in prefixes:
            continue

        if not os.path.isfile(os.path.join(args.input, basename)):
            continue

        ext_pair = ".bin" if ext == ".idx" else ".idx"
        assert os.path.isfile(
            os.path.join(args.input, prefix) + ext_pair
        ), f"ERROR: {ext_pair} file not provided for {os.path.join(args.input, prefix)}"

        prefixes.add(prefix)

    builder = None

    for prefix in sorted(prefixes):
        print_datetime(f"start processing file {prefix}")
        if builder is None:
            dataset = indexed_dataset.make_dataset(os.path.join(args.input, prefix), args.data_impl)

            if isinstance(dataset, indexed_dataset.MMapIndexedDataset):
                builder = indexed_dataset.MMapIndexedDatasetBuilder(
                    args.output_prefix + ".bin", dtype=dataset._index.dtype
                )
            else:
                builder = indexed_dataset.IndexedDatasetBuilder(args.output_prefix + ".bin", dtype=dataset.dtype)

            del dataset
        print_datetime(f"start merge file {prefix}")
        builder.merge_file_(os.path.join(args.input, prefix))
        print_datetime(f"end merge file {prefix}")

    print_datetime("start finalize")
    builder.finalize(args.output_prefix + ".idx")
    print_datetime("end finalize")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input", type=str, required=True, help="Path to directory containing all document files to merge"
    )
    group.add_argument("--data_impl", type=str, required=True, help="data_impl")

    group = parser.add_argument_group(title="output data")
    group.add_argument("--output-prefix", type=str, required=True, help="Path to binary output file without suffix")

    args = parser.parse_args()

    assert os.path.isdir(args.input), f"ERROR: {args.input} is not a directory or does not exist"

    assert os.path.isdir(
        os.path.dirname(args.output_prefix)
    ), f"ERROR: {os.path.dirname(args.output_prefix)} is not a directory or does not exist"

    main(args)
