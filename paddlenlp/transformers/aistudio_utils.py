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

from aistudio_sdk.hub import download


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
    download_kwargs = {}
    if revision is not None:
        download_kwargs["revision"] = revision
    if cache_dir is not None:
        download_kwargs["cache_dir"] = cache_dir
    res = download(
        repo_id=repo_id,
        filename=filename,
        **download_kwargs,
    )
    if "path" in res:
        return res["path"]
    else:
        if res["error_code"] == 10001:
            raise ValueError("Illegal argument error")
        elif res["error_code"] == 10002:
            raise UnauthorizedError(
                "Unauthorized Access. Please ensure that you have provided the AIStudio Access Token and you have access to the requested asset"
            )
        elif res["error_code"] == 12001:
            raise EntryNotFoundError(f"Cannot find the requested file '{filename}' in repo '{repo_id}'")
        else:
            raise Exception(f"Unknown error: {res}")
