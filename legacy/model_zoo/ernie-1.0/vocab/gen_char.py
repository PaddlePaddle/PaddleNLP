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

import os
import pickle
import sys
import time
from collections import defaultdict

input_path = sys.argv[1]
print(input_path)

char_dict = defaultdict(int)

file_paths = []
if os.path.isfile(input_path):
    file_paths.append(input_path)
else:
    for root, _, fs in os.walk(input_path):
        for f in fs:
            file_paths.append(os.path.join(root, f))

count = 0
s = time.time()
data_len = 0
for file_name in file_paths:
    print(f" > reading file {file_name}")
    with open(file_name, "r") as f:
        line = f.readline()
        while line:
            count += 1
            data_len += len(line.encode("utf-8"))
            for char in line:
                char_dict[char] += 1
            line = f.readline()
            if count % 10000 == 0:
                print(
                    f"processed doc {count}, char size: {len(char_dict)}, speed: {data_len/1024/1024/(time.time() - s)} MB/s"
                )
                with open("char_dict.txt", "w") as rf:
                    res = sorted(char_dict.items(), key=lambda x: -x[1])
                    for x in res:
                        k, v = x
                        rf.write(f"{k} {v}\n")

with open("char_dict.txt", "w") as f:
    res = sorted(char_dict.items(), key=lambda x: -x[1])
    for x in res:
        k, v = x
        f.write(f"{k} {v}\n")

with open("char_dict.pickle", "wb") as f:
    pickle.dump(char_dict, f)
