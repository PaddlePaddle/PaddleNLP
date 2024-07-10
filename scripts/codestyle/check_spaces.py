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

import re
import sys


def add_spaces_between_chinese_and_english(text):
    # 正则表达式匹配中文字符后紧跟英文字符或英文字符后紧跟中文字符的情况
    pattern = r"([\u4e00-\u9fa5])([a-zA-Z])|([a-zA-Z])([\u4e00-\u9fa5])"

    def replace_func(match):
        return match.group(1) + " " + match.group(2) if match.group(1) else match.group(3) + " " + match.group(4)

    return re.sub(pattern, replace_func, text)


def process_file(file_path):
    with open(file_path, "r+", encoding="utf-8") as file:
        content = file.read()
        new_content = add_spaces_between_chinese_and_english(content)
        if new_content != content:
            file.seek(0)
            file.write(new_content)
            file.truncate()
            print(f"Spaces added to {file_path}")


if __name__ == "__main__":
    for file_path in sys.argv[1:]:
        process_file(file_path)