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

from typing import Optional

from aistudio_sdk.file_download import model_file_download


class UnauthorizedError(Exception):
    pass


class EntryNotFoundError(Exception):
    pass


def _add_subfolder(weights_name: str, subfolder: Optional[str] = None) -> str:
    if subfolder is not None and subfolder != "":
        weights_name = "/".join([subfolder, weights_name])
    return weights_name


def aistudio_download(
    repo_id: str,
    filename: str = None,
    cache_dir: Optional[str] = None,
    subfolder: Optional[str] = "",
    revision: Optional[str] = None,
    **kwargs,
):
    if revision is None:
        revision = "master"
    filename = _add_subfolder(filename, subfolder)
    return model_file_download(
        repo_id=repo_id,
        file_path=filename,
        revision=revision,
        local_dir=cache_dir if cache_dir is not None else None
    )
