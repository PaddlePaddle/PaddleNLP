#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import multiprocessing
import os
import time
import warnings
from multiprocessing import Process

"""
Multi-process batch processing tool

This tool provides a multi-process batch processing method.
For example, multi-process batch download data, multi-process preprocessing data, etc.

The tool relies on executable shell commands or scripts. Its essence is to use Python's
multi-process library to create multiple processes, and call executable commands or
scripts through the os.system API.

Executable commands or scripts are passed in via a txt text file, organized by line.
For example, the following example is download, unzip and delete example.

batch_cmd.txt

wget http://xxxx.com/0.tar && tar -xf 0.tar && rm 0.tar
wget http://xxxx.com/1.tar && tar -xf 1.tar && rm 1.tar
...
wget http://xxxx.com/99.tar && tar -xf 99.tar && rm 99.tar

How to run:

python multiprocess_tool.py --num_proc 10 --shell_cmd_list_filename batch_cmd.txt

"""


def process_fn(cmd_list):
    for cmd in cmd_list:
        try:
            ret = os.system(cmd)
            if ret != 0:
                raise Exception(f"execute command: {cmd} failed.")
        except Exception as e:
            print(e)


def read_command(shell_cmd_list_filename):
    shell_cmd_list = []
    with open(shell_cmd_list_filename, "r") as f:
        for cmd in f:
            cmd = cmd.strip()
            shell_cmd_list.append(cmd)
    return shell_cmd_list


def parallel_process(cmd_list, nproc=20):
    if nproc > multiprocessing.cpu_count():
        warnings.warn(
            "The set number of processes exceeds the number of cpu cores, please confirm whether it is reasonable."
        )
    num_cmd = len(cmd_list)
    num_cmd_part = (num_cmd + nproc - 1) // nproc
    workers = []
    for i in range(min(nproc, num_cmd)):
        start = i * num_cmd_part
        end = min(start + num_cmd_part, num_cmd)
        p = Process(target=process_fn, args=(cmd_list[start:end],))
        workers.append(p)
        p.start()

    for p in workers:
        p.join()


def main(args):
    start = time.time()
    shell_cmd_list = read_command(args.shell_cmd_list_filename)
    parallel_process(shell_cmd_list, args.num_proc)
    end = time.time()
    print("Cost time: {:.2f}".format(end - start))


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="multi-process batch processing tool")
    parse.add_argument("--num_proc", type=int, default=20)
    parse.add_argument(
        "--shell_cmd_list_filename", type=str, help="a txt file contains shell command list to be execute."
    )
    args = parse.parse_args()
    main(args)
