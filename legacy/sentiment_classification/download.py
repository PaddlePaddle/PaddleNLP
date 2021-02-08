# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

"""
Download script, download dataset and pretrain models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import sys
import time
import hashlib
import tarfile
import requests

def usage():
    desc = ("\nDownload datasets and pretrained models for Sentiment Classification task.\n"
        "Usage:\n"
        "   1. python download.py dataset\n"
        "   2. python download.py model\n")
    print(desc)

def extract(fname, dir_path):
    """
    Extract tar.gz file
    """
    try:
        tar = tarfile.open(fname, "r:gz")
        file_names = tar.getnames()
        for file_name in file_names:
            tar.extract(file_name, dir_path)
            print(file_name)
        tar.close()
    except Exception as e:
        raise e

def download(url, filename):
    """
    Download file
    """
    retry = 0
    retry_limit = 3
    chunk_size = 4096
    while not os.path.exists(filename):
        if retry < retry_limit:
            retry += 1
        else:
            raise RuntimeError("Cannot download dataset ({0}) with retry {1} times.".
                format(url, retry_limit))
        try:
            start = time.time()
            size = 0
            res = requests.get(url, stream=True)
            filesize = int(res.headers['content-length'])
            if res.status_code == 200:
                print("[Filesize]: %0.2f MB" % (filesize / 1024 / 1024))
                # save by chunk
                with io.open(filename, "wb") as fout:
                    for chunk in res.iter_content(chunk_size=chunk_size):
                        if chunk:
                            fout.write(chunk)
                            size += len(chunk)
                            pr = '>' * int(size * 50 / filesize)
                            print('\r[Process ]: %s%.2f%%' % (pr, float(size / filesize*100)), end='')
            end = time.time()
            print("\n[CostTime]: %.2f s" % (end - start))
        except Exception as e:
            print(e)

def download_dataset(dir_path):
    BASE_URL = "https://baidu-nlp.bj.bcebos.com/"
    DATASET_NAME = "sentiment_classification-dataset-1.0.0.tar.gz"
    file_path = os.path.join(dir_path, DATASET_NAME)
    url = BASE_URL + DATASET_NAME
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # download dataset
    print("Downloading dataset: %s" % url)
    download(url, file_path)
    # extract dataset
    print("Extracting dataset: %s" % file_path)
    extract(file_path, dir_path)
    os.remove(file_path)

def download_model(dir_path):
    BASE_URL = "https://baidu-nlp.bj.bcebos.com/"
    MODEL_NAME = "sentiment_classification-1.0.0.tar.gz"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    url = BASE_URL + MODEL_NAME
    model_path = os.path.join(dir_path, MODEL_NAME)
    print("Downloading model: %s" % url)
    # download model
    download(url, model_path)
    # extract model.tar.gz
    print("Extracting model: %s" % model_path)
    extract(model_path, dir_path)
    os.remove(model_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        usage()
        sys.exit(1)
    
    if sys.argv[1] == "dataset":
        pwd = os.path.join(os.path.dirname(__file__), "./")
        download_dataset(pwd)
    elif sys.argv[1] == "model":
        pwd = os.path.join(os.path.dirname(__file__), "./models")
        download_model(pwd)
    else:
        usage()
