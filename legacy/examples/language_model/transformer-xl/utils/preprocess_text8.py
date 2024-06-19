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
import zipfile

if __name__ == "__main__":
    data = zipfile.ZipFile("text8.zip").extractall()
    data = open("text8", "r", encoding="utf-8").read()

    num_test_char = int(sys.argv[1])

    train_data = data[: -2 * num_test_char]
    valid_data = data[-2 * num_test_char : -num_test_char]
    test_data = data[-num_test_char:]

    for files, data in [("train.txt", train_data), ("valid.txt", valid_data), ("test.txt", test_data)]:
        data_str = " ".join(["_" if c == " " else c for c in data.strip()])
        with open(files, "w") as f:
            f.write(data_str)
        with open(files + ".raw", "w", encoding="utf-8") as fw:
            fw.write(data)
