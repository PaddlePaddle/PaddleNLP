# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

ilp_data_dir = "../ilp_data"
ilp_data_files = os.listdir(ilp_data_dir)
num_splits = len(ilp_data_files)

ilp_data = {}
common_keys = ["names", "shapes", "nparams", "qconfigs"]
for ilp_data_file in ilp_data_files:
    ilp_data_path = os.path.join(ilp_data_dir, ilp_data_file)
    ilp_data_part = paddle.load(ilp_data_path)
    if any(key not in ilp_data.keys() for key in common_keys):
        for key in common_keys:
            ilp_data[key] = ilp_data_part[key]
    for key in ["costs", "weights"]:
        start, end = ilp_data_file.split(".")[-2].split("-")
        index = int(start) // (int(end) - int(start) + 1)
        sub_tensors = paddle.split(ilp_data_part[key], num_splits, axis=0)
        if key in ilp_data.keys():
            ilp_data[key] = paddle.concat([ilp_data[key], sub_tensors[index]], axis=0)
        else:
            ilp_data[key] = sub_tensors[index]

merge_ilp_data_dir = os.path.join(ilp_data_dir, "merge")
if not os.path.exists(merge_ilp_data_dir):
    os.mkdir(merge_ilp_data_dir)
paddle.save(ilp_data, os.path.join(merge_ilp_data_dir, "llama2-7b.ilp.ranks-64.pth"))
