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


def get_repo_root(file_path):
    """通过给定文件路径找到仓库根目录"""
    current_dir = os.path.dirname(file_path)
    while not os.path.exists(os.path.join(current_dir, ".git")):
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        if parent_dir == current_dir:
            raise FileNotFoundError("Could not find .git directory")
        current_dir = parent_dir
    return current_dir


def find_dead_links(directory):
    # 正则表达式，用于匹配Markdown和reStructuredText中的链接
    markdown_link_pattern = r"\[([^\[\]]+)\]\(([^)]+)\)"  # 修改正则表达式以捕获链接文本
    rst_link_pattern = r"``([^`]+) <([^>]+)>`_"  # reStructuredText链接
    dead_links = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith((".md", ".rst")):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                        # 查找Markdown链接
                        markdown_matches = re.findall(markdown_link_pattern, content)
                        for link_text, match in markdown_matches:
                            if match.startswith(("http:", "https:")):
                                # 忽略外部链接
                                continue
                            elif "#" in match:
                                # 这是一个锚点链接，忽略文件系统检查
                                continue
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
            print("file path:", link[0], "- link text:", link[1], "- deal link:", link[2])
        print("Please check the above dead links and fix them.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        repo_root = get_repo_root(os.path.realpath(sys.argv[1]))
        process_file(repo_root)
