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
import re
import sys


def find_dead_links(file_path):
    # 正则表达式，用于匹配Markdown和reStructuredText中的链接
    markdown_link_pattern = r"\[([^\[\]]+)\]\(([^)]+)\)"  # 修改正则表达式以捕获链接文本
    rst_link_pattern = r"``([^`]+) <([^>]+)>`_"  # reStructuredText链接
    dead_links = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            print(f"Processing {file_path}")
            # 查找Markdown链接
            markdown_matches = re.findall(markdown_link_pattern, content)
            for link_text, match in markdown_matches:
                if not match.startswith(("http:", "https:")):
                    abs_path = os.path.abspath(os.path.join(os.path.dirname(file_path), match))
                    if not os.path.exists(abs_path):
                        dead_links.append((file_path, link_text, "Markdown Link: " + abs_path))

            # 查找reStructuredText链接
            rst_matches = re.findall(rst_link_pattern, content)
            for text, url in rst_matches:
                if not url.startswith(("http:", "https:")):
                    abs_path = os.path.abspath(os.path.join(os.path.dirname(file_path), url))
                    if not os.path.exists(abs_path):
                        dead_links.append((file_path, text, "reStructuredText Link: " + abs_path))

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return dead_links


def process_file(file_path):
    dead_links = find_dead_links(file_path)
    if len(dead_links) > 0:
        print("Dead links found in", file_path)
        for link in dead_links:
            print("\t", *link)
        return -1
    return 0


if __name__ == "__main__":
    for file_path in sys.argv[1:]:
        state = process_file(file_path)
