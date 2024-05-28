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


import json
import os
import re
import sys

import numpy as np


def analyze(model_item, log_file, res_log_file, device_num, bs, fp_item):
    with open(str(log_file), "r", encoding="utf8") as f:
        data = f.readlines()
    ips_lines = []
    for eachline in data:
        if "train_samples_per_second:" in eachline:
            ips = float(eachline.split("train_samples_per_second: ")[1].split()[0].replace(",", ""))
            print("----ips: ", ips)
            ips_lines.append(ips)
    print("----ips_lines: ", ips_lines)
    ips = np.round(np.mean(ips_lines), 3)
    ngpus = int(re.findall("\d+", device_num)[-1])
    print("----ips: ", ips, "ngpus", ngpus)
    ips *= ngpus
    run_mode = "DP"

    model_name = model_item + "_" + "bs" + str(bs) + "_" + fp_item + "_" + run_mode
    info = {
        "model_branch": os.getenv("model_branch"),
        "model_commit": os.getenv("model_commit"),
        "model_name": model_name,
        "batch_size": bs,
        "fp_item": fp_item,
        "run_mode": run_mode,
        "convergence_value": 0,
        "convergence_key": "",
        "ips": ips,
        "speed_unit": "sample/sec",
        "device_num": device_num,
        "model_run_time": os.getenv("model_run_time"),
        "frame_commit": os.getenv("frame_commit"),
        "frame_version": os.getenv("frame_version"),
    }
    json_info = json.dumps(info)
    print(json_info)
    with open(res_log_file, "w") as of:
        of.write(json_info)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage:" + sys.argv[0] + " model_item path/to/log/file path/to/res/log/file")
        sys.exit()

    model_item = sys.argv[1]
    log_file = sys.argv[2]
    res_log_file = sys.argv[3]
    device_num = sys.argv[4]
    bs = int(sys.argv[5])
    fp_item = sys.argv[6]

    analyze(model_item, log_file, res_log_file, device_num, bs, fp_item)
