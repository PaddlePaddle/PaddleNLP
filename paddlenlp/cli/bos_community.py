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
from __future__ import annotations

import multiprocessing
import os

from tqdm import tqdm

from paddlenlp.utils.log import logger

try:
    from baidubce.auth.bce_credentials import BceCredentials
    from baidubce.bce_client_configuration import BceClientConfiguration
    from baidubce.services.bos.bos_client import BosClient
except ImportError:
    logger.info("can not import baidubce module, please install it before running this command")


def upload_file(file_path: str, bos_file_path: str, bucket_name: str, bos_client: BosClient):
    """upload local file to bos

    Args:
        file_path (str): the path of local file
        bucket_name (str): the path of bos path
    """
    logger.info(f"start to upload file<{file_path}> to bos<{bucket_name}>")
    result = bos_client.put_super_obejct_from_file(
        bucket_name, bos_file_path, file_path, chunk_size=100, thread_num=multiprocessing.cpu_count()
    )
    if result:
        logger.info(f"Upload file <{file_path}> success!")
    else:
        logger.info(f"Upload file <{file_path}> fail!")


def get_files(local_path: str, origin_path: str | None = None) -> list[str]:
    """get all file path under the local file

    Args:
        local_path (str): the path of local directory

    Returns:
        list[str]: path of local file
    """
    origin_path = origin_path or local_path
    all_files = []
    for file in os.listdir(local_path):
        file_path = os.path.join(local_path, file)
        if os.path.isdir(file_path):
            all_files.extend(get_files(file_path, origin_path))
        else:
            bos_file_path = file_path.replace(origin_path, "")
            if bos_file_path.startswith("/"):
                bos_file_path = bos_file_path[1:]

            all_files.append((file_path, bos_file_path))

    return all_files


def bos_upload_handler(
    bos_path: str | None = None,
    local_path: str | None = None,
    bos_host: str | None = None,
):
    if bos_path is None:
        bos_path = "models/community"
        logger.info(
            f"can not detect src_path, so it will upload the content of current dir<{os.path.abspath(bos_path)}> to bos"
        )

    if local_path is None:
        local_path = "./"
        logger.info(
            f"can not detect local_path, so it will upload the content of current dir<{os.path.abspath(local_path)}> to bos"
        )

    # only load ak/sk from env
    access_key_id = os.getenv("bos_access_key_id", None)
    secret_access_key = os.getenv("bos_secret_access_key", None)

    if access_key_id is None or secret_access_key is None:
        raise ValueError(
            "Please set environment variables of `bos_access_key_id`, `bos_secret_access_key`, before uploading !!!"
        )

    if bos_host is None:
        bos_host = os.getenv("bos_host", "paddlenlp.bj.bcebos.com")
    logger.info(f"bos host: {bos_host}")

    # 1. init bos_client
    config = BceClientConfiguration(credentials=BceCredentials(access_key_id, secret_access_key), endpoint=bos_host)

    bos_client = BosClient(config)

    # 2. upload local file
    if os.path.isfile(local_path):
        raise ValueError(f"local_path<{local_path}> can not be file, it must be directory")

    for file_path, bos_file_path in tqdm(get_files(local_path)):
        upload_file(file_path=file_path, bos_file_path=bos_file_path, bucket_name=bos_path, bos_client=bos_client)
