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
import subprocess
from typing import List

from git import Diff, Repo
from typer import Typer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)

from paddlenlp.utils.log import logger  # noqa: E402


def file_in_dir(file_path: str, dir_path: str) -> bool:
    """check whether the `file_path` is the file under `dir_path`

    Args:
        file_path (str): the path of file
        dir_path (str): the path of dir
    """
    assert os.path.isfile(file_path)
    assert os.path.isdir(dir_path)
    return os.path.abspath(dir_path) in os.path.abspath(file_path)


def get_changed_files() -> List[str]:
    """get the changed file locally.

    Returns:
        List[str]: list of path of changed files
    """
    repo = Repo(ROOT_DIR)
    logger.warning(
        "you should run the command: `git fetch upstream` to fetch the "
        "latest commit info of upstream.")

    develop_commit = repo.commit("upstream/develop")
    diff_indexes: List[Diff] = repo.head.commit.diff(develop_commit)

    all_files = set()

    diffs = []
    for diff_index in diff_indexes:
        diffs.append(diff_index.change_type)
        all_files.add(diff_index.a_path)
        all_files.add(diff_index.b_path)

    valid_files = []
    for file in all_files:
        file_path = os.path.join(ROOT_DIR, file)
        if os.path.exists(file_path):
            valid_files.append(file)

    return valid_files


def get_target_changed_files(file_suffix: str = '.py') -> List[str]:
    """get the target changed files with file_suffix.

    Args:
        file_suffix (str, optional): the suffix of different type of file,
            which can be `.md`, `.py`, `.yml`. Defaults to '.py'.

    Returns:
        List[str]: the different type of file path
    """
    files = get_changed_files()
    files = [file for file in files if file.endswith(file_suffix)]
    return files


app = Typer()


@app.command(help='use isort to lint the importings')
def isort():
    logger.info("start to use `isort` to lint the imports in python files")
    python_files = get_target_changed_files()

    for file in python_files:
        logger.debug(f"find file: {file}")

    file_string = " ".join(python_files)
    subprocess.call(f'cd {ROOT_DIR} && isort {file_string}', shell=True)


@app.command(help='use flake8 to format the python files')
def flake8():
    python_files = get_target_changed_files()
    file_string = " ".join(python_files)
    subprocess.call(f'cd {ROOT_DIR} && flake8 {file_string}', shell=True)


@app.command(help='run pylint')
def pylint():
    python_files = get_target_changed_files()
    file_string = " ".join(python_files)
    subprocess.call(f'cd {ROOT_DIR} && pylint {file_string}', shell=True)


if __name__ == "__main__":
    app()
