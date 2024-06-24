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

import os
import shutil
import time

import paddle
import requests
from ppfleetx.utils.log import logger
from tqdm import tqdm

DOWNLOAD_RETRY_LIMIT = 3


def is_url(path):
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith("http://") or path.startswith("https://")


def _map_path(url, root_dir):
    # parse path after download under root_dir
    fname = os.path.split(url)[-1]
    fpath = fname
    return os.path.join(root_dir, fpath)


def cached_path(url_or_path, cache_dir=None):
    if cache_dir is None:
        cache_dir = "~/.cache/ppfleetx/"

    cache_dir = os.path.expanduser(cache_dir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    if is_url(url_or_path):
        path = _map_path(url_or_path, cache_dir)
        url = url_or_path
    else:
        path = url_or_path
        url = None

    if os.path.exists(path):
        logger.info(f"Found {os.path.split(path)[-1]} in cache_dir: {cache_dir}.")
        return path

    download(url, path)
    return path


def _download(url, fullname):
    """
    Download from url, save to path.
    url (str): download url
    path (str): download to given path
    """
    retry_cnt = 0

    while not os.path.exists(fullname):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            raise RuntimeError("Download from {} failed. " "Retry limit reached".format(url))

        logger.info("Downloading {}".format(url))

        try:
            req = requests.get(url, stream=True)
        except Exception as e:  # requests.exceptions.ConnectionError
            logger.info("Downloading {} failed {} times with exception {}".format(url, retry_cnt + 1, str(e)))
            time.sleep(1)
            continue

        if req.status_code != 200:
            raise RuntimeError("Downloading from {} failed with code " "{}!".format(url, req.status_code))

        # For protecting download interupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = fullname + "_tmp"
        total_size = req.headers.get("content-length")
        with open(tmp_fullname, "wb") as f:
            if total_size:
                with tqdm(total=(int(total_size) + 1023) // 1024) as pbar:
                    for chunk in req.iter_content(chunk_size=1024):
                        f.write(chunk)
                        pbar.update(1)
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, fullname)

    return fullname


def download(url, path):
    local_rank = 0
    world_size = 1
    if paddle.base.core.is_compiled_with_dist() and paddle.distributed.get_world_size() > 1:
        local_rank = paddle.distributed.ParallelEnv().dev_id
        world_size = paddle.distributed.get_world_size()
    if world_size > 1 and local_rank != 0:
        while not os.path.exists(path):
            time.sleep(1)
    else:
        _download(url, path)
