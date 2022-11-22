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
from typing import List
from git import Repo, Diff
from typer import Typer
import subprocess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)


def get_changed_files() -> List[str]:
    repo = Repo(ROOT_DIR)

    # Your last commit of the dev branch
    develop_commit = repo.commit("origin/develop")
    all_files = set()

    diff_indexes: List[Diff] = repo.head.commit.diff(develop_commit)

    all_files = set()
    for diff_index in diff_indexes:
        all_files.add(diff_index.a_path)
        all_files.add(diff_index.b_path)

    valid_files = []
    for file in all_files:
        file_path = os.path.join(ROOT_DIR, file)
        if os.path.exists(file_path):
            valid_files.append(file)

    return valid_files


def get_target_changed_files(file_suffix: str = '.py') -> List[str]:
    files = get_changed_files()
    files = [file for file in files if file.endswith(file_suffix)]
    return files


app = Typer()


@app.command(help='use isort to lint the importings')
def isort():
    python_files = get_target_changed_files()
    file_string = " ".join(python_files)
    subprocess.call(f'cd {ROOT_DIR} && isort {file_string}', shell=True)


@app.command(help='use flake8 to format the python files')
def flake8():
    python_files = get_target_changed_files()
    file_string = " ".join(python_files)
    subprocess.call(f'cd {ROOT_DIR} && flake8 {file_string}', shell=True)


if __name__ == "__main__":
    app()
