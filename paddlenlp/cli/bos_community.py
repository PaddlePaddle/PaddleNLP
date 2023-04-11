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
import sys

from baidubce.auth.bce_credentials import BceCredentials
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.services.bos.bos_client import BosClient

bos_config = {
    "bucket": "models",
    "bos_host": "paddlenlp.bj.bcebos.com",
}


bos_host = str(bos_config["bos_host"])
bos_bucket = str(bos_config["bucket"])

access_key_id = os.getenv("bos_access_key_id", None)
secret_access_key = os.getenv("bos_secret_access_key", None)
if access_key_id is None or secret_access_key is None:
    raise ValueError(
        "Please set environment variables of  bos_access_key_id, bos_secret_access_key, before uploading !!!"
    )


def upload_to_bos_from_raw(raw, name, category="test"):
    b_config = BceClientConfiguration(credentials=BceCredentials(access_key_id, secret_access_key), endpoint=bos_host)
    bos_client = BosClient(b_config)
    bos_client.put_object_from_string(bos_bucket, "%s/%s" % (category, name), raw)
    url = "https://paddlenlp.bj.bcebos.com/%s/%s/%s" % (bos_bucket, category, name)
    return url


def multi_upload_to_bos(filename, name, category):
    b_config = BceClientConfiguration(credentials=BceCredentials(access_key_id, secret_access_key), endpoint=bos_host)
    bos_client = BosClient(b_config)
    # init multi-upload
    key = "%s/%s" % (category, name)
    bucket_name = bos_bucket
    upload_id = bos_client.initiate_multipart_upload(bucket_name, key).upload_id

    left_size = os.path.getsize(filename)
    offset = 0
    part_number = 1
    part_list = []
    while left_size > 0:
        part_size = 3 * 1024 * 1024 * 1024
        if left_size < part_size:
            part_size = left_size
        response = bos_client.upload_part_from_file(
            bucket_name, key, upload_id, part_number, part_size, filename, offset
        )
        left_size -= part_size
        offset += part_size
        # your should store every part number and etag to invoke complete multi-upload
        part_list.append({"partNumber": part_number, "eTag": response.metadata.etag})
        part_number += 1
    bos_client.complete_multipart_upload(bucket_name, key, upload_id, part_list)
    url = "https://paddlenlp.bj.bcebos.com/%s/%s/%s" % (bos_bucket, category, name)
    return url


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python bos_community.py organization/model local_dir")
        sys.exit(1)

    organization = sys.argv[1]
    local_dir = sys.argv[2]

    for filename in os.listdir(local_dir):
        name = os.path.split(filename)[-1]
        if name == "bos.log":
            continue
        filename = os.path.join(local_dir, filename)
        left_size = os.path.getsize(filename)
        print(f"Uploading to {organization}/{name}, size: {left_size}")
        if left_size >= 5 * 1024 * 1024 * 1024:
            url = multi_upload_to_bos(filename, name, category=f"community/{organization}")
        else:
            with open(filename, "rb") as fp:
                url = upload_to_bos_from_raw(raw=fp.read(), name=name, category=f"community/{organization}")
        print(f"Done: {url}")
