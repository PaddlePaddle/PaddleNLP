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

import sys

import yaml


def parse_benchmark(filein, fileout):
    with open(filein, "r") as fin:
        res = yaml.load(fin)
        del_list = []
        for key in res["DAG"].keys():
            if "call" in key:
                del_list.append(key)
        for key in del_list:
            del res["DAG"][key]
    with open(fileout, "w") as fout:
        yaml.dump(res, fout, default_flow_style=False)


if __name__ == "__main__":
    filein = sys.argv[1]
    fileout = sys.argv[2]
    parse_benchmark(filein, fileout)
